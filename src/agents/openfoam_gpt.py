"""
Agent③ OpenFOAMGPT Agent
EnrichedContext を受け取り、RAG コンテキストを注入した LLM プロンプトで
OpenFOAM ケースファイルを生成・実行し、失敗時に自己修正する
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ..config import Settings
from ..case_applier import CaseApplier, copy_zero_orig_if_needed
from ..case_validator import CaseValidator
from ..error_fixer import apply_rule_based_fixes
from ..llm_client import LLMClient
from ..mesh.cylinder_2d_ogrid import generate as generate_cylinder_2d_ogrid
from ..models import (
    SimulationSpec, EnrichedContext, GenerationResult, CaseArtifacts
)
from ..runner import OpenFOAMRunner
from ..monitor import SolverMonitor

console = Console()

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
MAX_RETRIES = 3

OPENFOAM_GPT_SYSTEM = """あなたはOpenFOAMのシミュレーションエンジニアです。
RAGで取得した参考例を活用し、物理的に正確なOpenFOAMケースファイルを生成します。
- OpenFOAM v2512 形式を厳守する
- 境界条件は物理的に妥当な値を設定する
- 数値スキームは解析タイプ（定常/非定常、乱流/層流）に合わせて選択する
- コードブロック記号(```)は絶対に出力しない
"""


class OpenFOAMGPTAgent:
    """Agent③: ファイル生成 → 実行 → 自己修正ループエージェント。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.runner = OpenFOAMRunner(settings)
        self.applier = CaseApplier()
        self.validator = CaseValidator()
        self.jinja = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def run(
        self,
        context: EnrichedContext,
        output_dir: str,
        convergence_threshold: float = 1e-4,
    ) -> CaseArtifacts:
        """
        EnrichedContext からケースを生成・実行し CaseArtifacts を返す。
        """
        spec = context.spec

        # ── Step 1: ケース生成 ────────────────────────────────────────
        console.print(Rule("[bold cyan]ケースファイルを生成中[/bold cyan]"))
        gen_result = self._generate_case(context, output_dir)
        case_dir = gen_result.output_path
        console.print(f"  出力先: [cyan]{case_dir}[/cyan] ({len(gen_result.files_created)} ファイル)")

        artifacts = CaseArtifacts(
            case_dir=case_dir,
            spec=spec,
            generation_result=gen_result,
            log_files={},
        )

        # ── Step 2: blockMesh（自己修正ループ）────────────────────────
        if context.reference_mesh_prebuilt:
            console.print(Rule("[bold cyan]blockMesh 実行[/bold cyan]"))
            console.print("  [dim]事前メッシュ使用 — blockMesh をスキップ[/dim]")
            artifacts.block_mesh_success = True
        else:
            console.print(Rule("[bold cyan]blockMesh 実行[/bold cyan]"))
            bm_result, bm_retries = self._run_with_self_correction(
                case_dir=case_dir,
                command="blockMesh",
                run_fn=lambda: self.runner.run_block_mesh(case_dir),
                fix_fn=lambda err, content: self._fix_blockmesh(err, case_dir, context),
            )
            artifacts.block_mesh_success = bm_result.success
            artifacts.block_mesh_retries = bm_retries
            artifacts.log_files["blockMesh"] = bm_result.log_file or ""

            if not bm_result.success:
                console.print(Panel(
                    f"[red]blockMesh が {MAX_RETRIES} 回試行後も失敗しました[/red]",
                    border_style="red",
                ))
                return artifacts

            console.print("[bold green]  ✓ メッシュ生成完了[/bold green]")

        # ── Step 3a: snappyHexMesh（STL指定時のみ）────────────────────
        if spec.case_type in ("snappy_external", "snappy_2d") and spec.stl_path:
            console.print(Rule("[bold cyan]snappyHexMesh 実行[/bold cyan]"))

            # surfaceFeatureExtract
            feat_result = self.runner.run_surface_feature_extract(case_dir)
            artifacts.log_files["surfaceFeatureExtract"] = feat_result.log_file or ""
            if not feat_result.success:
                console.print("[yellow]  ⚠ surfaceFeatureExtract 失敗（続行）[/yellow]")
            else:
                console.print("[green]  ✓ フィーチャーエッジ抽出完了[/green]")

            # snappyHexMesh（自己修正なし: 失敗時はログを報告）
            snappy_result = self.runner.run_snappy_hex_mesh(case_dir)
            artifacts.log_files["snappyHexMesh"] = snappy_result.log_file or ""
            if not snappy_result.success:
                console.print(Panel(
                    "[red]snappyHexMesh が失敗しました。ログを確認してください。[/red]\n"
                    f"ログ: {snappy_result.log_file}",
                    border_style="red",
                ))
                return artifacts
            console.print("[bold green]  ✓ snappyHexMesh 完了[/bold green]")

        # ── Step 3: checkMesh ─────────────────────────────────────────
        cm = self.runner.run_check_mesh(case_dir)
        artifacts.log_files["checkMesh"] = cm.log_file or ""
        console.print("[green]  ✓ メッシュ品質確認完了[/green]" if cm.success else
                      "[yellow]  ⚠ checkMesh に警告あり（続行）[/yellow]")

        # blockMesh 後の patch 整合性チェック
        post_mesh_issues = self.validator.validate(
            Path(case_dir), spec, after_blockmesh=True
        )
        post_errors = [i for i in post_mesh_issues if i.severity == "error"]
        if post_errors:
            for issue in post_errors[:3]:
                console.print(f"[yellow]  ⚠ {issue.check}: {issue.message}[/yellow]")

        # ── Step 3b: 初期化 ───────────────────────────────────────────
        is_karman = spec.phenomenon == "karman_vortex_shedding"
        if not spec.steady_state and is_karman and spec.case_type == "cylinder_2d_ogrid":
            self._apply_karman_seed(case_dir, context)
        elif not spec.steady_state and (
            spec.case_type in (
                "snappy_2d", "snappy_external", "external_snappy", "cylinder_2d_ogrid",
            )
            or context.reference_case_id
        ):
            console.print("[cyan]  → potentialFoam で速度場を初期化中...[/cyan]")
            pot = self.runner.run_potential_foam(case_dir)
            if pot.returncode == 0:
                console.print("[green]  ✓ potentialFoam 初期化完了[/green]")
            else:
                console.print("[yellow]  ⚠ potentialFoam 失敗（初期フィールドのままで続行）[/yellow]")

        # ── Step 4: ソルバー実行（自己修正ループ）────────────────────
        console.print(Rule("[bold cyan]ソルバー実行[/bold cyan]"))
        log_file = str(Path(case_dir) / f"log.{spec.solver}")
        monitor = SolverMonitor(log_file=log_file, convergence_threshold=convergence_threshold)

        # reference case 使用時は solver を controlDict に合わせる
        cd_path = Path(case_dir) / "system" / "controlDict"
        if cd_path.exists() and context.reference_case_id:
            import re as _re
            m = _re.search(r"application\s+(\w+)\s*;", cd_path.read_text())
            if m:
                spec.solver = m.group(1)

        solver_result, solver_retries = self._run_with_self_correction(
            case_dir=case_dir,
            command=spec.solver,
            run_fn=lambda: monitor.watch(
                solver_result_fn=lambda: self.runner.run_solver(case_dir, spec.solver)
            ),
            fix_fn=lambda err, _: self._fix_solver_settings(err, case_dir, context),
        )
        artifacts.solver_success = solver_result.success
        artifacts.solver_retries = solver_retries
        artifacts.log_files[spec.solver] = solver_result.log_file or ""

        # 最終収束状態を取得
        final_status = monitor.parse_log()
        artifacts.final_residuals = final_status.residuals

        if spec.steady_state:
            # 定常: 残差閾値を下回ったかどうかで収束判定
            artifacts.converged = final_status.converged
            if artifacts.converged:
                console.print("[bold green]  ✓ 収束完了[/bold green]")
            elif not solver_result.success:
                console.print(Panel(
                    f"[red]ソルバーが {MAX_RETRIES} 回試行後も失敗しました[/red]",
                    border_style="red",
                ))
            else:
                console.print("[yellow]  ⚠ endTime に達しましたが残差閾値未達[/yellow]")
        else:
            # 非定常 (pimpleFoam 等): endTime 到達 = 正常終了
            artifacts.converged = solver_result.success
            if solver_result.success:
                console.print("[bold green]  ✓ 非定常計算完了 (endTime 到達)[/bold green]")
            else:
                console.print(Panel(
                    f"[red]ソルバーが {MAX_RETRIES} 回試行後も失敗しました[/red]",
                    border_style="red",
                ))

        return artifacts

    # ──────────────────────────────────────────────────────────────────
    # ケースファイル生成
    # ──────────────────────────────────────────────────────────────────

    def _generate_case(self, context: EnrichedContext, output_dir: str) -> GenerationResult:
        """EnrichedContext からケースファイル一式を生成する。"""
        spec = context.spec
        case_name = re.sub(r'[^\w-]', '_', f"{spec.solver}_{spec.case_type}")
        case_path = Path(output_dir) / case_name
        self._reset_case_dir(case_path)

        files_created: list[str] = []

        # ── 主経路: 参照ケースを丸ごと適用 ──────────────────────────
        if context.reference_files and context.reference_case_id:
            files_created = self.applier.apply(context, case_path)
            copy_zero_orig_if_needed(case_path)

            # 事前メッシュケースは blockMeshDict を生成しない
            if not context.reference_mesh_prebuilt and (
                spec.case_type == "cylinder_2d_ogrid"
                or not (case_path / "system" / "blockMeshDict").exists()
            ):
                bmd = self._render_blockmesh_template(context)
                (case_path / "system" / "blockMeshDict").write_text(bmd)
                if "system/blockMeshDict" not in files_created:
                    files_created.append("system/blockMeshDict")

            pre_issues = self.validator.validate(case_path, spec)
            for issue in pre_issues:
                if issue.severity == "error":
                    console.print(f"[yellow]  ⚠ 検証: {issue.check}: {issue.message}[/yellow]")

            if spec.case_type in ("snappy_external", "snappy_2d") and spec.stl_path:
                self._setup_snappy_files(case_path, context, files_created)

            return GenerationResult(
                output_path=str(case_path),
                case_type=spec.case_type,
                files_created=files_created,
            )

        # ── フォールバック: Jinja テンプレート ──────────────────────
        console.print("  [dim]参照ケースなし — Jinja テンプレートで生成[/dim]")
        jinja_context = self._build_jinja_context(context)

        bmd_content = self._render_blockmesh_template(context)
        (case_path / "system" / "blockMeshDict").write_text(bmd_content)
        files_created.append("system/blockMeshDict")

        template_map = {
            "system/controlDict": "system/controlDict.j2",
            "system/fvSchemes": "system/fvSchemes.j2",
            "system/fvSolution": "system/fvSolution.j2",
            "constant/turbulenceProperties": "constant/turbulenceProperties.j2",
            "constant/transportProperties": "constant/transportProperties.j2",
            "0/U": "0/U.j2",
            "0/p": "0/p.j2",
        }
        if spec.turbulence_model != "laminar":
            template_map.update({
                "0/k": "0/k.j2",
                "0/omega": "0/omega.j2",
                "0/nut": "0/nut.j2",
            })
        for out_rel, tmpl_name in template_map.items():
            try:
                content = self.jinja.get_template(tmpl_name).render(**jinja_context)
                (case_path / out_rel).write_text(content)
                files_created.append(out_rel)
            except Exception:
                pass

        if spec.case_type in ("snappy_external", "snappy_2d") and spec.stl_path:
            self._setup_snappy_files(case_path, context, files_created)

        return GenerationResult(
            output_path=str(case_path),
            case_type=spec.case_type,
            files_created=files_created,
        )

    def _reset_case_dir(self, case_path: Path) -> None:
        """再生成時に参照ケース由来の古いファイルを残さない。"""
        if case_path.exists():
            for name in case_path.iterdir():
                if name.is_dir():
                    shutil.rmtree(name)
                else:
                    name.unlink()
        else:
            case_path.mkdir(parents=True, exist_ok=True)
        (case_path / "0").mkdir(exist_ok=True)
        (case_path / "constant").mkdir(exist_ok=True)
        (case_path / "system").mkdir(exist_ok=True)

    def _render_blockmesh_template(self, context: EnrichedContext) -> str:
        """blockMeshDict テンプレートをパラメータで埋めてレンダリングする。"""
        spec = context.spec
        # Agent① の spec.mesh_template を最優先。RAG の mesh_template_name は補助的に使う
        raw_name = spec.mesh_template or context.mesh_template_name or "box_channel_2d"
        # 旧テンプレート名との後方互換
        _TMPL_COMPAT = {
            "box_external": "box_channel_2d",
            "box_internal": "box_channel_3d",
            "box_2d":       "box_channel_2d",
        }
        template_name = _TMPL_COMPAT.get(raw_name, raw_name)
        params = context.mesh_params_suggestion or {}

        # O-グリッド円柱 2D: Python ジェネレータで直接生成 (Jinja2 不使用)
        if template_name == "ogrid_cylinder_2d":
            spec = context.spec
            r = (spec.characteristic_length or 0.1) / 2.0
            lchar = r * 2  # = characteristic_length
            return generate_cylinder_2d_ogrid(
                cx=0.0, cy=0.0, r=r,
                ring_factor=2.0,
                x_in=round(-lchar * 8,  4),   # 上流 8D
                x_out=round(lchar * 20, 4),    # 下流 20D
                y_min=round(-lchar * 10, 4),   # ±10D (ブロッケージ 5%)
                y_max=round(lchar * 10,  4),
                z_min=0.0, z_max=0.01,
                n_r=15, n_t=20, n_up=40, n_down=80, n_lat=30,
                # gr_r=0.05 (極細セル ~0.5mm) は dt=4ms で Co=7 → 爆発
                # gr_r=0.5  (最小セル ~2.3mm) は dt=3ms で Co=0.4 → 安定
                gr_r=0.5, gr_up=0.1, gr_down=10.0, gr_lat=0.1,
            )

        # テンプレートに応じたデフォルトパラメータ
        if template_name == "box_channel_2d":
            x_min = params.get("x_min", -5.0)
            defaults = {
                "x_min": x_min,
                "x_max": x_min + params.get("lx", 20.0),
                "ly":    params.get("ly", 5.0),
                "depth": params.get("depth", 0.1),
                "nx":    params.get("nx", 40),
                "ny":    params.get("ny", 20),
            }
        elif template_name == "box_channel_3d":
            defaults = {
                "x_min": params.get("x_min", 0.0),
                "x_max": params.get("x_min", 0.0) + params.get("lx", 10.0),
                "ly":    params.get("ly", 1.0),
                "lz":    params.get("lz", 1.0),
                "nx":    params.get("nx", 60),
                "ny":    params.get("ny", 20),
                "nz":    params.get("nz", 20),
            }
        elif template_name == "box_2d":
            x_min = params.get("x_min", -5.0)
            defaults = {
                "x_min": x_min,
                "x_max": x_min + params.get("lx", 20.0),
                "ly":    params.get("ly", 5.0),
                "depth": 0.1,
                "nx":    params.get("nx", 40),
                "ny":    params.get("ny", 20),
            }
        elif template_name == "box_internal":
            gy = params.get("grading_y", 4.0)
            defaults = {
                "lx": params.get("lx", 10.0),
                "ly": params.get("ly", 1.0),
                "lz": params.get("lz", 1.0),
                "nx": params.get("nx", 80),
                "ny": params.get("ny", 20),
                "nz": params.get("nz", 20),
                "grading_y": gy,
                "grading_y_inv": round(1.0 / gy, 4),
            }
        elif template_name == "box_snappy_2d":
            # 2D snappyHexMesh 用: z 方向は固定 0〜0.01m, front/back = empty
            # ブロッケージ比 < 5% を確保するため Y 幅は ±10D 以上にする
            stl_bbox = params.get("stl_bbox", {})
            cx = stl_bbox.get("cx", 0.0)
            cy = stl_bbox.get("cy", 0.0)
            lchar = stl_bbox.get("lchar", context.spec.characteristic_length or 0.1)
            defaults = {
                "stl_name": params.get("stl_name", "geometry.stl"),
                "x_min": round(cx - lchar * 8,  4),   # 上流 8D
                "x_max": round(cx + lchar * 20, 4),   # 下流 20D（渦列の発達に十分）
                "y_min": round(cy - lchar * 10, 4),   # ±10D → ブロッケージ = D/20D = 5%
                "y_max": round(cy + lchar * 10, 4),
                # RAG 提案値は粗すぎるため固定値を使う
                "nx": 120,
                "ny": 70,
            }
        elif template_name == "box_snappy":
            # snappyHexMesh 用背景メッシュ: STLのバウンディングボックスから自動計算
            stl_bbox = params.get("stl_bbox", {})
            cx = stl_bbox.get("cx", 0.0)
            cy = stl_bbox.get("cy", 0.0)
            cz = stl_bbox.get("cz", 0.0)
            lchar = stl_bbox.get("lchar", context.spec.characteristic_length)
            scale = params.get("domain_scale", 10.0)
            half = lchar * scale / 2.0
            # 流れ方向(x)は非対称: 上流2倍・下流4倍
            defaults = {
                "stl_name": params.get("stl_name", "geometry.stl"),
                "x_min": round(cx - half * 0.6, 4),
                "x_max": round(cx + half * 1.4, 4),
                "y_min": round(cy - half, 4),
                "y_max": round(cy + half, 4),
                "z_min": round(cz - half, 4),
                "z_max": round(cz + half, 4),
                "nx": params.get("nx", 30),
                "ny": params.get("ny", 20),
                "nz": params.get("nz", 20),
                "domain_scale": scale,
            }
        else:  # フォールバック: box_channel_2d
            template_name = "box_channel_2d"
            x_min = params.get("x_min", -5.0)
            defaults = {
                "x_min": x_min,
                "x_max": x_min + params.get("lx", 20.0),
                "ly":    params.get("ly", 5.0),
                "depth": params.get("depth", 0.1),
                "nx":    params.get("nx", 40),
                "ny":    params.get("ny", 20),
            }

        tmpl_path = f"system/blockMeshDict/{template_name}.j2"
        try:
            return self.jinja.get_template(tmpl_path).render(**defaults)
        except Exception as e:
            console.print(f"[yellow]  blockMeshDict テンプレートエラー: {e}[/yellow]")
            # フォールバック: 最もシンプルな 2D チャンネル
            return self.jinja.get_template("system/blockMeshDict/box_channel_2d.j2").render(
                x_min=-5.0, x_max=15.0, ly=5.0, depth=0.1, nx=40, ny=20
            )

    def _build_jinja_context(self, context: EnrichedContext) -> dict:
        """Jinja2 テンプレートに渡すコンテキストを構築する。"""
        spec = context.spec
        has_wall = spec.case_type in (
            "channel_2d", "channel_3d", "external_snappy", "snappy_external",
            "snappy_2d", "heat_transfer", "internal_flow", "cylinder_2d_ogrid",
        )
        # snappyHexMesh が STL 名からパッチを自動生成するため、ステム名を渡す
        snappy_object_name = Path(spec.stl_path).stem if spec.stl_path else "object"
        is_snappy_2d = (spec.case_type == "snappy_2d")
        is_ogrid_2d = (spec.case_type == "cylinder_2d_ogrid")
        is_karman_ogrid = (
            is_ogrid_2d and spec.phenomenon == "karman_vortex_shedding"
        )
        purge_write = 0 if spec.steady_state else 5
        karman_r = (getattr(spec, "characteristic_length", 1.0) or 1.0) / 2.0
        wake_x0 = round(karman_r * 1.05, 4)
        wake_x1 = round(karman_r * 6.0, 4)
        wake_y = round(karman_r * 2.0, 4)
        perturb_vy = round(spec.inlet_velocity * 0.05, 6)
        depth = 0.01

        if spec.steady_state:
            end_time, delta_t, write_interval = 1000, 1, 100
        else:
            # 非定常: 流れが特性長を10往復する程度の時間を目安に設定
            char_len = getattr(spec, "characteristic_length", 1.0) or 1.0
            flow_through = char_len / max(spec.inlet_velocity, 0.01)
            end_time = round(flow_through * 10, 4)   # 10 flow-through times
            write_interval = round(flow_through / 10, 5)  # 100 スナップショット

            if is_karman_ogrid:
                # Strouhal ~0.2 @ Re~100 → 25 周期分 + 周期あたり 20 フレーム
                st = 0.2
                shed_period = char_len / max(st * spec.inlet_velocity, 1e-6)
                end_time = round(shed_period * 25, 2)
                write_interval = round(shed_period / 20, 4)
                purge_write = 0
                min_cell_approx = char_len * 0.007
                delta_t = round(
                    min_cell_approx * 0.1 / max(spec.inlet_velocity * 2, 0.01), 6
                )
            elif spec.case_type == "cylinder_2d_ogrid":
                # pimpleFoam + 固定 dt (timeStep writeControl)
                # 最小セルサイズ: 接線方向 ≈ π*r/n_t_total = π*0.5*lchar/(8*20) ≈ 0.0098*lchar
                # 径方向 (gr_r=0.5, n_r=15): ≈ 0.007*lchar
                # → 最小セル ≈ 0.007*lchar, Co=0.1, U_max=2*U_inf
                min_cell_approx = char_len * 0.007
                delta_t = round(min_cell_approx * 0.1 / max(spec.inlet_velocity * 2, 0.01), 6)
            elif (spec.solver == "icoFoam" and spec.case_type == "snappy_2d"):
                # icoFoam は adjustTimeStep を使わないため、固定 dt を安全側に設定
                min_cell_approx = char_len * 0.023
                delta_t = round(min_cell_approx * 0.3 / max(spec.inlet_velocity * 3, 0.01), 6)
            elif spec.case_type in ("external_snappy", "snappy_external", "snappy_2d"):
                # snappyHexMesh は表面付近に極細セルを作るため初期 dt を小さく
                # adjustTimeStep が自動的に最適値に上げていく
                delta_t = round(flow_through / 50000, 8)
            else:
                delta_t = round(flow_through / 500, 6)  # CFL < 1 になる初期値

        return {
            "solver": spec.solver,
            "case_type": spec.case_type,
            "turbulence_model": spec.turbulence_model,
            "steady_state": spec.steady_state,
            "dimensions": spec.dimensions if hasattr(spec, "dimensions") else 3,
            "description": spec.description,
            "inlet_velocity": spec.inlet_velocity,
            "has_wall": has_wall,
            "snappy_object_name": snappy_object_name,
            "is_snappy_2d": is_snappy_2d,
            "is_ogrid_2d": is_ogrid_2d,
            "is_karman_ogrid": is_karman_ogrid,
            "phenomenon": spec.phenomenon,
            "purge_write": purge_write,
            "wake_x0": wake_x0,
            "wake_x1": wake_x1,
            "wake_y": wake_y,
            "perturb_vy": perturb_vy,
            "depth": depth,
            "end_time": end_time,
            "delta_t": delta_t,
            "write_interval": write_interval,
        }

    def _apply_karman_seed(self, case_dir: str, context: EnrichedContext) -> None:
        """カルマン渦: 対称な potentialFoam の代わりに後流へ摂動を与える。"""
        case_path = Path(case_dir)
        jinja_context = self._build_jinja_context(context)
        content = self.jinja.get_template("system/setFieldsDict.j2").render(**jinja_context)
        (case_path / "system" / "setFieldsDict").write_text(content)
        result = self.runner.run_set_fields(case_dir)
        if result.returncode == 0:
            console.print("[green]  ✓ 後流摂動を付与（カルマン渦分岐用）[/green]")
        else:
            console.print("[yellow]  ⚠ setFields 失敗（数値ノイズに依存して続行）[/yellow]")

    def _setup_snappy_files(self, case_path: Path, context: EnrichedContext, files_created: list) -> None:
        """snappyHexMesh に必要なファイルを生成・コピーする。
        - constant/triSurface/ に STL をコピー
        - STL の bbox を解析して blockMeshDict を更新
        - system/snappyHexMeshDict を生成
        - system/surfaceFeatureExtractDict を生成
        """
        import shutil
        from pathlib import Path as _P

        spec = context.spec
        stl_src = _P(spec.stl_path)
        stl_name = stl_src.name

        # constant/triSurface/ に STL をコピー
        tri_dir = case_path / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        stl_dst = tri_dir / stl_name
        shutil.copy2(stl_src, stl_dst)
        console.print(f"  [green]STLをコピー: {stl_name} → constant/triSurface/[/green]")
        files_created.append(f"constant/triSurface/{stl_name}")

        # STL のバウンディングボックスを解析
        bbox = self._analyze_stl_bbox(stl_dst)
        console.print(
            f"  [dim]STL bbox: x=[{bbox['x_min']:.3f},{bbox['x_max']:.3f}] "
            f"y=[{bbox['y_min']:.3f},{bbox['y_max']:.3f}] "
            f"z=[{bbox['z_min']:.3f},{bbox['z_max']:.3f}][/dim]"
        )

        # mesh_params_suggestion に bbox 情報を追加して blockMeshDict を再生成
        params = dict(context.mesh_params_suggestion or {})
        lchar = max(
            bbox["x_max"] - bbox["x_min"],
            bbox["y_max"] - bbox["y_min"],
            bbox["z_max"] - bbox["z_min"],
        )
        params["stl_bbox"] = {
            "cx": (bbox["x_min"] + bbox["x_max"]) / 2,
            "cy": (bbox["y_min"] + bbox["y_max"]) / 2,
            "cz": (bbox["z_min"] + bbox["z_max"]) / 2,
            "lchar": lchar,
        }
        params["stl_name"] = stl_name
        params["domain_scale"] = 10.0
        # context.mesh_params_suggestion を一時的に書き換えてレンダリング
        orig_params = context.mesh_params_suggestion
        orig_tmpl = context.mesh_template_name
        context.mesh_params_suggestion = params
        context.mesh_template_name = "box_snappy"
        bmd_content = self._render_blockmesh_template(context)
        context.mesh_params_suggestion = orig_params
        context.mesh_template_name = orig_tmpl

        (case_path / "system" / "blockMeshDict").write_text(bmd_content)
        console.print("  [green]背景メッシュ (blockMeshDict) を STL bbox に合わせて更新しました[/green]")

        # snappyHexMeshDict
        stl_solid_name = stl_name.rsplit(".", 1)[0]
        # locationInMesh: 円柱から上流側にズラした流体領域内の点
        # 上流 4D の位置（円柱中心から十分離れており、STL の外側）
        loc_x = round(params["stl_bbox"]["cx"] - lchar * 4.0, 4)
        loc_y = round(params["stl_bbox"]["cy"] + lchar * 0.1, 4)
        if spec.case_type == "snappy_2d":
            # 2D: ドメインは z=[0, 0.01] 固定なので中心 0.005 を使う
            loc_z = 0.005
        else:
            loc_z = round(params["stl_bbox"]["cz"] + lchar * 0.3, 4)
        snappy_ctx = {
            "stl_name": stl_name,
            "stl_solid_name": stl_solid_name,
            # 2D ケース: level 5 は z 方向の極薄セルを生成して発散の原因になるため抑制
            "feature_level": 2 if spec.case_type != "snappy_2d" else 1,
            "surface_min_level": 3 if spec.case_type != "snappy_2d" else 2,
            "surface_max_level": 5 if spec.case_type != "snappy_2d" else 3,
            "location_x": loc_x,
            "location_y": loc_y,
            "location_z": loc_z,
        }
        snappy_content = self.jinja.get_template("system/snappyHexMeshDict.j2").render(**snappy_ctx)
        (case_path / "system" / "snappyHexMeshDict").write_text(snappy_content)
        files_created.append("system/snappyHexMeshDict")
        console.print("  [green]snappyHexMeshDict を生成しました[/green]")

        # surfaceFeatureExtractDict
        feat_content = self.jinja.get_template("system/surfaceFeatureExtractDict.j2").render(
            stl_name=stl_name
        )
        (case_path / "system" / "surfaceFeatureExtractDict").write_text(feat_content)
        files_created.append("system/surfaceFeatureExtractDict")
        console.print("  [green]surfaceFeatureExtractDict を生成しました[/green]")

    @staticmethod
    def _analyze_stl_bbox(stl_path: Path) -> dict:
        """STL ファイルのバウンディングボックスを解析する（ASCII/バイナリ両対応）。"""
        import struct

        xs, ys, zs = [], [], []

        # ASCII STL かバイナリ STL かを判定
        try:
            text = stl_path.read_text(errors="ignore")
            if text.strip().startswith("solid"):
                # ASCII STL: vertex 行を解析
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("vertex"):
                        parts = line.split()
                        if len(parts) == 4:
                            xs.append(float(parts[1]))
                            ys.append(float(parts[2]))
                            zs.append(float(parts[3]))
        except Exception:
            pass

        if not xs:
            # バイナリ STL
            try:
                data = stl_path.read_bytes()
                # 84バイト目以降がトライアングルデータ (50バイト毎)
                n_tri = struct.unpack_from("<I", data, 80)[0]
                for i in range(n_tri):
                    offset = 84 + i * 50 + 12  # normalをスキップ
                    for _ in range(3):
                        x, y, z = struct.unpack_from("<fff", data, offset)
                        xs.append(x); ys.append(y); zs.append(z)
                        offset += 12
            except Exception:
                pass

        if not xs:
            # フォールバック: デフォルト bbox
            return {"x_min": -0.5, "x_max": 0.5, "y_min": -0.5, "y_max": 0.5, "z_min": -0.5, "z_max": 0.5}

        return {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "z_min": min(zs), "z_max": max(zs),
        }

    def _enhance_fvschemes(self, case_path: Path, context: EnrichedContext) -> bool:
        """RAG の推薦スキームを使って fvSchemes を LLM で強化する。"""
        try:
            current = (case_path / "system" / "fvSchemes").read_text()
            prompt = (
                f"以下は現在のfvSchemesです。RAGで取得した推薦スキームを参考に改善してください。\n"
                f"コードブロック(```)なしで、fvSchemesの内容のみ出力してください。\n\n"
                f"現在のfvSchemes:\n{current}\n\n"
                f"RAG推薦スキーム（参考）:\n{context.recommended_schemes}"
            )
            enhanced = self.llm.chat(prompt, system=OPENFOAM_GPT_SYSTEM)
            enhanced = self.llm._strip_code_fences(enhanced)
            if "divSchemes" in enhanced and "gradSchemes" in enhanced:
                (case_path / "system" / "fvSchemes").write_text(enhanced)
                return True
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────────────────────────
    # 自己修正ループ
    # ──────────────────────────────────────────────────────────────────

    def _run_with_self_correction(
        self, case_dir: str, command: str,
        run_fn, fix_fn,
    ) -> tuple:
        """
        コマンドを実行し、失敗したら LLM で修正してリトライする。

        Returns:
            (RunResult, retries_count)
        """
        for attempt in range(1, MAX_RETRIES + 1):
            result = run_fn()
            if result.success:
                return result, attempt - 1

            console.print(f"[yellow]  {command} 失敗 (試行 {attempt}/{MAX_RETRIES})[/yellow]")

            if attempt < MAX_RETRIES:
                error_context = (result.stdout or "")[-2000:] + (result.stderr or "")[-500:]
                console.print(f"  [cyan]LLM がエラーを解析・修正中...[/cyan]")
                fix_fn(error_context, result.stdout or "")

        return result, MAX_RETRIES - 1

    def _fix_blockmesh(self, error_msg: str, case_dir: str, context: EnrichedContext) -> None:
        """blockMeshDict のエラーを LLM で修正する。"""
        bmd_path = Path(case_dir) / "system" / "blockMeshDict"
        current = bmd_path.read_text() if bmd_path.exists() else ""

        prompt = (
            f"以下のblockMeshDictでblockMeshを実行したところエラーが発生しました。\n"
            f"エラーを修正したblockMeshDictを出力してください（コードブロックなし）。\n\n"
            f"エラーメッセージ:\n{error_msg}\n\n"
            f"現在のblockMeshDict:\n{current}"
        )
        fixed = self.llm.chat(prompt, system=OPENFOAM_GPT_SYSTEM)
        fixed = self.llm._strip_code_fences(fixed)
        if "FoamFile" in fixed or "vertices" in fixed:
            bmd_path.write_text(fixed)
            console.print("  blockMeshDict を修正しました")

    def _fix_solver_settings(self, error_msg: str, case_dir: str, context: EnrichedContext) -> None:
        """ソルバーエラー時の修正戦略:
        1. ルールベース修正を試みる（高速・確実）
        2. 修正できなかった場合のみテンプレート再生成にフォールバック
        """
        case_path = Path(case_dir)
        # まずルールベース修正を試みる
        rule_fixed = apply_rule_based_fixes(case_dir, error_msg)
        if rule_fixed:
            console.print("  [green]ルールベース修正を適用しました（テンプレート再生成をスキップ）[/green]")
            return
        # ルールで解決できなければテンプレートから再生成
        console.print("  [dim]テンプレートから system/ と 0/ を再生成します[/dim]")
        self._regenerate_system_dir(case_path, context)
        self._regenerate_zero_dir(case_path, context)

    def _regenerate_system_dir(self, case_path: Path, context: EnrichedContext) -> None:
        """system/ を再生成。参照ケースがあれば再適用、なければ Jinja。"""
        if context.reference_files:
            self.applier.apply(context, case_path)
            return
        jinja_context = self._build_jinja_context(context)
        for fname in ["fvSchemes", "fvSolution", "controlDict"]:
            try:
                content = self.jinja.get_template(f"system/{fname}.j2").render(**jinja_context)
                (case_path / "system" / fname).write_text(content)
                console.print(f"  [dim]system/{fname} をテンプレートから再生成しました[/dim]")
            except Exception:
                pass

    def _regenerate_zero_dir(self, case_path: Path, context: EnrichedContext) -> None:
        """0/ を再生成。参照ケースがあれば再適用、なければ Jinja。"""
        if context.reference_files:
            for rel, content in context.reference_files.items():
                if rel.startswith("0/"):
                    (case_path / rel).write_text(content)
            self.applier._substitute_parameters(case_path, context.spec, [])
            return
        jinja_context = self._build_jinja_context(context)
        for fname in ["U", "p", "k", "omega", "nut"]:
            try:
                content = self.jinja.get_template(f"0/{fname}.j2").render(**jinja_context)
                (case_path / "0" / fname).write_text(content)
                console.print(f"  [dim]0/{fname} をテンプレートから再生成しました[/dim]")
            except Exception:
                pass

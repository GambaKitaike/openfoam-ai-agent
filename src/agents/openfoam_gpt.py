"""
Agent③ OpenFOAMGPT Agent
EnrichedContext を受け取り、RAG コンテキストを注入した LLM プロンプトで
OpenFOAM ケースファイルを生成・実行し、失敗時に自己修正する
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..case_builder.file_generators import FileGenerator
from ..case_builder.pipeline import CaseBuildPipeline
from ..case_builder.reference_path import apply_reference_fast_path
from ..case_builder.snappy_generators import (
    build_snappy_hex_mesh_dict,
    build_surface_feature_extract_dict,
    render_snappy_block_mesh_dict,
)
from ..case_builder.policy import read_patch_names
from ..config import Settings
from ..case_applier import CaseApplier
from ..case_validator import CaseValidator
from ..error_fixer import apply_rule_based_fixes
from ..llm_client import LLMClient
from ..models import (
    EnrichedContext, GenerationResult, CaseArtifacts, ReferenceMatch,
)
from ..runner import OpenFOAMRunner
from ..case_runtime import read_end_time
from ..monitor import SolverMonitor
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

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
        self.file_gen = FileGenerator(self.llm)
        self.pipeline = CaseBuildPipeline(settings)
        self._guidance_context: EnrichedContext | None = None
        self._guidance_fn = None

    def run(
        self,
        context: EnrichedContext,
        output_dir: str,
        convergence_threshold: float = 1e-4,
        reference_match: ReferenceMatch | None = None,
        parallel: bool = False,
        n_procs: int = 4,
        file_guidance_fn=None,
    ) -> CaseArtifacts:
        """
        EnrichedContext からケースを生成・実行し CaseArtifacts を返す。
        """
        spec = context.spec

        self._guidance_context = context
        self._guidance_fn = file_guidance_fn
        if file_guidance_fn:
            self.pipeline = CaseBuildPipeline(
                self.settings,
                guidance_fn=lambda rel, s, patches: file_guidance_fn(
                    rel, s, context, patch_names=patches
                ),
            )

        # ── Step 1: ケース生成 ────────────────────────────────────────
        console.print(Rule("[bold cyan]ケースファイルを生成中[/bold cyan]"))
        gen_result = self._generate_case(context, output_dir, reference_match)
        case_dir = gen_result.output_path
        console.print(f"  出力先: [cyan]{case_dir}[/cyan] ({len(gen_result.files_created)} ファイル)")

        artifacts = CaseArtifacts(
            case_dir=case_dir,
            spec=spec,
            generation_result=gen_result,
            log_files={},
        )

        # ── Step 2: blockMesh（自己修正ループ）────────────────────────
        if gen_result.mesh_built:
            console.print(Rule("[bold cyan]blockMesh 実行[/bold cyan]"))
            console.print("  [dim]段階的生成で blockMesh 済み — スキップ[/dim]")
            artifacts.block_mesh_success = True
        elif context.reference_mesh_prebuilt:
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
            if self.pipeline.apply_karman_seed(case_dir):
                console.print("[green]  ✓ 後流摂動を付与（カルマン渦分岐用）[/green]")
            else:
                console.print("[yellow]  ⚠ setFields 失敗（数値ノイズに依存して続行）[/yellow]")
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
        end_time = read_end_time(case_dir)
        monitor = SolverMonitor(
            log_file=log_file,
            convergence_threshold=convergence_threshold,
            steady_state=spec.steady_state,
            end_time=end_time,
        )

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
                solver_result_fn=lambda: self.runner.run_solver(
                    case_dir, spec.solver, parallel=parallel, n_procs=n_procs
                )
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

    def _generate_case(
        self,
        context: EnrichedContext,
        output_dir: str,
        reference_match: ReferenceMatch | None = None,
    ) -> GenerationResult:
        """EnrichedContext からケースファイル一式を生成する（fast path / staged）。"""
        spec = context.spec
        is_snappy = spec.case_type in ("snappy_external", "snappy_2d") and spec.stl_path
        run_mesh_in_pipeline = not is_snappy and not context.reference_mesh_prebuilt

        if reference_match and reference_match.use_fast_path:
            try:
                console.print("  [cyan]経路 A: 参照ケース fast path[/cyan]")
                _state, gen = apply_reference_fast_path(reference_match, output_dir)
                if is_snappy:
                    self._setup_snappy_files(Path(gen.output_path), context, gen.files_created)
                pre_issues = self.validator.validate(Path(gen.output_path), spec)
                for issue in pre_issues:
                    if issue.severity == "error":
                        console.print(
                            f"[yellow]  ⚠ 検証: {issue.check}: {issue.message}[/yellow]"
                        )
                return gen
            except Exception as exc:
                console.print(
                    f"[yellow]  fast path 失敗 → staged にフォールバック: {exc}[/yellow]"
                )

        console.print("  [cyan]経路 B: 段階的ケース生成[/cyan]")
        _state, gen = self.pipeline.run(context, output_dir, run_mesh=run_mesh_in_pipeline)
        if is_snappy:
            self._setup_snappy_files(Path(gen.output_path), context, gen.files_created)
            gen.mesh_built = False
        return gen

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
        cx = params["stl_bbox"]["cx"]
        cy = params["stl_bbox"]["cy"]
        is_2d = spec.case_type == "snappy_2d"
        if is_2d:
            x_min = round(cx - lchar * 8, 4)
            x_max = round(cx + lchar * 20, 4)
            y_min = round(cy - lchar * 10, 4)
            y_max = round(cy + lchar * 10, 4)
        else:
            half = lchar * params.get("domain_scale", 10.0) / 2.0
            x_min = round(cx - half * 0.6, 4)
            x_max = round(cx + half * 1.4, 4)
            y_min = round(cy - half, 4)
            y_max = round(cy + half, 4)

        bmd_content = render_snappy_block_mesh_dict(
            stl_name=stl_name,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            is_2d=is_2d,
        )
        (case_path / "system" / "blockMeshDict").write_text(bmd_content)
        console.print("  [green]背景メッシュ (blockMeshDict) を STL bbox に合わせて更新しました[/green]")

        stl_solid_name = stl_name.rsplit(".", 1)[0]
        loc_x = round(cx - lchar * 4.0, 4)
        loc_y = round(cy + lchar * 0.1, 4)
        loc_z = 0.005 if is_2d else round(params["stl_bbox"]["cz"] + lchar * 0.3, 4)
        snappy_content = build_snappy_hex_mesh_dict(
            stl_name=stl_name,
            stl_solid_name=stl_solid_name,
            location_x=loc_x,
            location_y=loc_y,
            location_z=loc_z,
            spec=spec,
        )
        (case_path / "system" / "snappyHexMeshDict").write_text(snappy_content)
        files_created.append("system/snappyHexMeshDict")
        console.print("  [green]snappyHexMeshDict を生成しました[/green]")

        feat_content = build_surface_feature_extract_dict(stl_name)
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
        console.print("  [dim]ビルダーから system/ と 0/ を再生成します[/dim]")
        self._regenerate_system_dir(case_path, context)
        self._regenerate_zero_dir(case_path, context)

    def _regenerate_system_dir(self, case_path: Path, context: EnrichedContext) -> None:
        """system/ を再生成。参照ケースがあれば再適用、なければ決定的ビルダー。"""
        if context.reference_files:
            self.applier.apply(context, case_path)
            return
        for rel in ("system/fvSchemes", "system/fvSolution", "system/controlDict"):
            content = self.file_gen.generate(rel, context)
            (case_path / rel).write_text(content)
            console.print(f"  [dim]{rel} を再生成しました[/dim]")

    def _regenerate_zero_dir(self, case_path: Path, context: EnrichedContext) -> None:
        """0/ を再生成。参照ケースがあれば再適用、なければ決定的ビルダー。"""
        if context.reference_files:
            for rel, content in context.reference_files.items():
                if rel.startswith("0/"):
                    (case_path / rel).write_text(content)
            self.applier._substitute_parameters(case_path, context.spec, [])
            return
        patch_names = read_patch_names(str(case_path)) or []
        for rel in ("0/U", "0/p"):
            content = self.file_gen.generate(rel, context, patch_names)
            (case_path / rel).write_text(content)
            console.print(f"  [dim]{rel} を再生成しました[/dim]")

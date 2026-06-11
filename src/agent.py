"""
OpenFOAM AI Agent - コアエージェントクラス
全パイプライン（ファイル生成 → メッシュ → 計算 → 後処理）を統括する
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .config import Settings
from .llm_client import LLMClient
from .case_generator import CaseGenerator
from .runner import OpenFOAMRunner
from .monitor import SolverMonitor
from .postprocess import PostProcessor
from .models import GenerationResult

console = Console()

# blockMesh 失敗時の最大リトライ回数
MAX_BLOCKMESH_RETRIES = 3


class OpenFOAMAgent:
    """
    自然言語の説明からOpenFOAMの解析を全自動実行するエージェント。

    パイプライン:
      1. LLM で解析仕様を構造化
      2. LLM で blockMeshDict を生成
      3. ケースファイルを書き出し
      4. blockMesh 実行（失敗時はLLMが自動修正してリトライ）
      5. checkMesh でメッシュ品質確認
      6. ソルバー実行（収束モニタリング付き）
      7. 後処理・ParaView 起動
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.generator = CaseGenerator(settings)
        self.runner = OpenFOAMRunner(settings)
        self.postprocessor = PostProcessor(settings)

    # ------------------------------------------------------------------
    # パブリックAPI
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        description: str,
        output_dir: str,
        open_paraview: bool = True,
        convergence_threshold: float = 1e-4,
    ) -> dict:
        """
        自然言語から ParaView 起動まで全工程を実行する。

        Args:
            description: 解析内容の自然言語説明
            output_dir: 出力ディレクトリ
            open_paraview: 解析完了後にParaViewを起動するか
            convergence_threshold: 収束判定閾値

        Returns:
            dict: 各ステップの実行結果
        """
        results = {}

        # ── Step 1: 解析仕様の構造化 ──────────────────────────────────
        console.print(Rule("[bold cyan]Step 1/6  解析仕様を解析中[/bold cyan]"))
        spec = self.llm.parse_analysis_spec(description)
        console.print(f"  ソルバー       : [bold]{spec.solver}[/bold]")
        console.print(f"  解析タイプ     : [bold]{spec.case_type}[/bold]")
        console.print(f"  乱流モデル     : [bold]{spec.turbulence_model}[/bold]")
        console.print(f"  定常/非定常    : [bold]{'定常' if spec.steady_state else '非定常'}[/bold]")
        results["spec"] = spec

        # ── Step 2: blockMeshDict の生成 ──────────────────────────────
        console.print(Rule("[bold cyan]Step 2/6  blockMeshDict を生成中[/bold cyan]"))
        block_mesh_dict_content = self.llm.generate_block_mesh_dict(spec)
        console.print("  blockMeshDict を生成しました")

        # ── Step 3: ケースファイルの書き出し ──────────────────────────
        console.print(Rule("[bold cyan]Step 3/6  ケースファイルを生成中[/bold cyan]"))
        gen_result = self.generator.generate(
            spec=spec,
            output_dir=output_dir,
            block_mesh_dict_content=block_mesh_dict_content,
        )
        case_dir = gen_result.output_path
        console.print(f"  出力先: [cyan]{case_dir}[/cyan]")
        console.print(f"  生成ファイル: {len(gen_result.files_created)} 件")
        results["generation"] = gen_result

        # ── Step 4: blockMesh 実行（失敗時は自動修正リトライ）──────────
        console.print(Rule("[bold cyan]Step 4/6  メッシュ生成 (blockMesh)[/bold cyan]"))
        bm_result = self._run_block_mesh_with_retry(
            case_dir=case_dir,
            initial_dict=block_mesh_dict_content,
        )
        results["block_mesh"] = bm_result

        if not bm_result.success:
            console.print(Panel(
                f"[red]blockMesh が失敗しました（{MAX_BLOCKMESH_RETRIES}回リトライ後）\n"
                f"ログ: {bm_result.log_file}[/red]",
                border_style="red",
            ))
            return results

        console.print("[bold green]  ✓ メッシュ生成完了[/bold green]")

        # ── Step 5: checkMesh ──────────────────────────────────────────
        console.print(Rule("[bold cyan]Step 5/6  ソルバー実行[/bold cyan]"))
        cm_result = self.runner.run_check_mesh(case_dir)
        results["check_mesh"] = cm_result
        if cm_result.success:
            console.print("[green]  ✓ メッシュ品質 OK[/green]")
        else:
            console.print("[yellow]  ⚠ checkMesh に問題があります（続行します）[/yellow]")

        # ソルバー実行（リアルタイム収束モニタリング）
        from .case_runtime import read_end_time

        log_file = str(Path(case_dir) / f"log.{spec.solver}")
        monitor = SolverMonitor(
            log_file=log_file,
            convergence_threshold=convergence_threshold,
            steady_state=spec.steady_state,
            end_time=read_end_time(case_dir),
        )

        solver_result = monitor.watch(
            solver_result_fn=lambda: self.runner.run_solver(case_dir, spec.solver)
        )
        results["solver"] = solver_result

        console.print("\n" + monitor.summary())

        if not solver_result.success:
            console.print(Panel(
                f"[red]ソルバーが異常終了しました\nログ: {solver_result.log_file}[/red]",
                border_style="red",
            ))
            return results

        console.print("[bold green]  ✓ 計算完了[/bold green]")

        # ── Step 6: 後処理・ParaView ───────────────────────────────────
        console.print(Rule("[bold cyan]Step 6/6  後処理・ParaView 起動[/bold cyan]"))
        post_results = self.postprocessor.run(case_dir, open_paraview=open_paraview)
        results["postprocess"] = post_results
        self.postprocessor.print_result_summary(case_dir)

        console.print(Panel(
            "[bold green]全工程が完了しました！[/bold green]\n"
            f"ケースディレクトリ: [cyan]{case_dir}[/cyan]",
            border_style="green",
        ))

        return results

    def generate(self, description: str, output_dir: str) -> GenerationResult:
        """ファイル生成のみ行う（blockMesh・ソルバーは実行しない）。"""
        spec = self.llm.parse_analysis_spec(description)
        block_mesh_dict_content = self.llm.generate_block_mesh_dict(spec)
        return self.generator.generate(
            spec=spec,
            output_dir=output_dir,
            block_mesh_dict_content=block_mesh_dict_content,
        )

    def review(self, case_dir: str) -> str:
        """既存のケースをレビューする。"""
        case_path = Path(case_dir)
        if not case_path.exists():
            return f"[red]エラー: {case_dir} が見つかりません[/red]"
        case_files = self._read_case_files(case_path)
        return self.llm.review_case(case_files)

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _run_block_mesh_with_retry(self, case_dir: str, initial_dict: str):
        """
        blockMesh を実行し、失敗した場合は LLM でエラーを修正してリトライする。
        """
        current_dict = initial_dict
        block_mesh_file = Path(case_dir) / "system" / "blockMeshDict"

        for attempt in range(1, MAX_BLOCKMESH_RETRIES + 1):
            result = self.runner.run_block_mesh(case_dir)

            if result.success:
                return result

            console.print(f"[yellow]  blockMesh 失敗 (試行 {attempt}/{MAX_BLOCKMESH_RETRIES})[/yellow]")

            if attempt < MAX_BLOCKMESH_RETRIES:
                console.print("  [cyan]LLM がエラーを修正中...[/cyan]")
                error_context = result.stdout[-2000:] + result.stderr[-500:]
                current_dict = self.llm.fix_block_mesh_error(current_dict, error_context)
                block_mesh_file.write_text(current_dict)
                console.print("  blockMeshDict を修正しました。リトライします...")

        return result

    def _read_case_files(self, case_path: Path) -> dict[str, str]:
        """OpenFOAMケースの主要ファイルを読み込む。"""
        files = {}
        important_paths = [
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/blockMeshDict",
            "constant/turbulenceProperties",
            "constant/transportProperties",
        ]
        for rel_path in important_paths:
            full_path = case_path / rel_path
            if full_path.exists():
                files[rel_path] = full_path.read_text()
        return files

"""
OpenFOAM AI Agent - メインエントリポイント
4-Agent Pipeline: Pre-processing → RAG → OpenFOAMGPT → Post-processing
"""
import sys

import typer
from rich.console import Console
from rich.panel import Panel

from .orchestrator import OpenFOAMOrchestrator
from .config import Settings

app = typer.Typer(
    help="OpenFOAMの解析ファイルを自動生成・実行するAIエージェント (4-Agent Pipeline)",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    description: str = typer.Argument(..., help="解析の内容を日本語で説明してください"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="出力先ディレクトリ"),
    threshold: float = typer.Option(1e-4, "--threshold", "-t", help="収束判定の残差閾値"),
    stl_file: str = typer.Option("", "--stl", "-s", help="物体形状のSTLファイルパス（snappyHexMesh用）"),
    interactive: bool = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="未指定パラメータを対話で確認 (デフォルト: TTYなら対話)",
    ),
):
    """
    【フルパイプライン】自然言語から後処理まで4エージェントが全工程を自動実行します。

    例: python -m src.main run "円柱周りの定常乱流解析、流入速度10m/s"
    例: python -m src.main run "円柱周りの外部流れ" --stl /path/to/cylinder.stl
    """
    settings = Settings()
    orchestrator = OpenFOAMOrchestrator(settings=settings)
    use_interactive = interactive if interactive is not None else sys.stdin.isatty()
    orchestrator.run(
        description=description,
        output_dir=output_dir,
        convergence_threshold=threshold,
        stl_path=stl_file,
        interactive=use_interactive,
    )


@app.command()
def build_index(
    no_web: bool = typer.Option(False, "--no-web", help="Webスクレイピングをスキップ"),
    skip_enrich: bool = typer.Option(False, "--skip-enrich", help="LLM意図メタデータ生成をスキップ"),
    enrich_only: bool = typer.Option(False, "--enrich-only", help="インデックス化せず enrich のみ"),
    force_enrich: bool = typer.Option(False, "--force-enrich", help="キャッシュを無視して LLM 再生成"),
):
    """
    【RAG構築】OpenFOAMチュートリアルとWebドキュメントをインデックス化します（初回のみ）。

    例: python -m src.main build-index
    """
    from pathlib import Path
    from .rag.indexer import OpenFOAMIndexer

    settings = Settings()
    db_path = Path(__file__).parent.parent / "knowledge_base" / "chroma_db"
    console.print(Panel.fit(
        "[bold cyan]RAG インデックス構築[/bold cyan]\n"
        "OpenFOAMチュートリアル + Webドキュメントをベクトルインデックス化します",
        border_style="cyan",
    ))
    console.print(f"保存先: [cyan]{db_path}[/cyan]\n")

    indexer = OpenFOAMIndexer(
        db_path=str(db_path),
        openai_api_key=settings.openai_api_key,
    )
    stats = indexer.build(
        include_web=not no_web,
        skip_enrich=skip_enrich,
        enrich_only=enrich_only,
        force_enrich=force_enrich,
    )
    console.print(
        f"\n[bold green]完了![/bold green] "
        f"インデックス化: {stats.get('cases', 0)} ケース "
        f"(スキップ {stats.get('skipped', 0)})"
    )
    if not skip_enrich:
        console.print(
            f"  intent: 新規 {stats.get('enriched', 0)}, "
            f"キャッシュ {stats.get('cached', 0)}, 失敗 {stats.get('intent_failed', 0)}"
        )


@app.command()
def check(
    case_dir: str = typer.Argument(..., help="チェックするOpenFOAMケースのディレクトリ"),
):
    """
    【レビュー】既存のOpenFOAMケースをAIがレビューします。

    例: python -m src.main check ./output/simpleFoam_external_flow
    """
    from pathlib import Path
    from .llm_client import LLMClient

    settings = Settings()
    llm = LLMClient(settings)
    case_path = Path(case_dir)
    if not case_path.exists():
        console.print(f"[red]エラー: {case_dir} が見つかりません[/red]")
        raise typer.Exit(1)

    files = {}
    for rel in ["system/controlDict", "system/fvSchemes", "system/fvSolution",
                "system/blockMeshDict", "constant/turbulenceProperties"]:
        p = case_path / rel
        if p.exists():
            files[rel] = p.read_text()

    with console.status("[bold green]AIがケースをレビュー中...[/bold green]"):
        report = llm.review_case(files)

    console.print(Panel(report, title="[bold]AIレビュー結果[/bold]", border_style="cyan"))


if __name__ == "__main__":
    app()

"""
OpenFOAM AI Agent - メインエントリポイント
4-Agent Pipeline: Pre-processing → RAG → OpenFOAMGPT → Post-processing
"""
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    _ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT))
    __package__ = "src"

import typer
from rich.console import Console
from rich.panel import Panel

from .orchestrator import OpenFOAMOrchestrator
from .config import Settings
from .agent_dialogue import BUILTIN_SCENARIOS

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
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="mpirun による並列ソルバー実行 (decomposePar → reconstructPar)",
    ),
    n_procs: int = typer.Option(
        4,
        "--np",
        min=2,
        help="並列プロセス数 (--parallel 時)",
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="短時間デモ設定 (カルマン渦: 5周期)。--periods 未指定時のみ",
    ),
    periods: int | None = typer.Option(
        None,
        "--periods",
        min=1,
        help="カルマン渦の放出周期数 (本番=25, デモ=5)。指定時は --demo より優先",
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
        parallel=parallel,
        n_procs=n_procs,
        demo=demo,
        periods=periods,
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


@app.command("test-agents")
def test_agents(
    description: str = typer.Argument(
        "",
        help="テストする自然言語入力（空なら内蔵シナリオを実行）",
    ),
    scenario: str = typer.Option(
        "",
        "--scenario",
        "-s",
        help="内蔵シナリオ: karman | channel_conflict | channel_laminar",
    ),
    all_scenarios: bool = typer.Option(
        False,
        "--all",
        help="内蔵シナリオをすべて実行",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive/--no-interactive",
        help="Agent② レビューをユーザー確認付きで実行",
    ),
    skip_match: bool = typer.Option(
        False,
        "--skip-match",
        help="Agent② → Agent③ の Reference Match をスキップ",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Agent① の LLM extract をスキップ（Agent 間ループのみテスト）",
    ),
):
    """
    【Agent 通信テスト】Agent①↔Agent② の内部ループと Reference Match を記録（ソルバー実行なし）。

    例:
      python -m src.main test-agents --all
      python -m src.main test-agents -s channel_conflict
      python -m src.main test-agents "2D円柱 Re=100 カルマン渦"
    """
    settings = Settings()
    orchestrator = OpenFOAMOrchestrator(settings=settings)

    builtin = BUILTIN_SCENARIOS

    if all_scenarios:
        for name, text in builtin.items():
            console.print(f"\n[bold]{'=' * 60}[/bold]")
            console.print(f"[bold]Scenario: {name}[/bold]\n")
            orchestrator.test_agent_dialogue(
                text,
                interactive=interactive,
                include_reference_match=not skip_match,
                offline=offline,
                scenario=name,
            )
        return

    if scenario:
        if scenario not in builtin:
            console.print(
                f"[red]未知のシナリオ: {scenario}[/red] "
                f"(利用可能: {', '.join(builtin)})"
            )
            raise typer.Exit(1)
        text = builtin[scenario]
        scenario_key = scenario
    elif description:
        text = description
        scenario_key = ""
    else:
        text = builtin["channel_conflict"]
        scenario_key = "channel_conflict"

    orchestrator.test_agent_dialogue(
        text,
        interactive=interactive,
        include_reference_match=not skip_match,
        offline=offline,
        scenario=scenario_key,
    )


@app.command("continue-run")
def continue_run(
    case_dir: str = typer.Argument(..., help="続行する OpenFOAM ケースディレクトリ"),
    end_time: float = typer.Option(..., "--end-time", "-e", help="新しい endTime [s]"),
    write_interval: float = typer.Option(
        None, "--write-interval", "-w", help="writeInterval [s]（省略時は変更なし）"
    ),
    n_procs: int = typer.Option(4, "--np", min=2, help="MPI プロセス数"),
    no_vtk: bool = typer.Option(False, "--no-vtk", help="foamToVTK をスキップ"),
):
    """
    既存ケースを latestTime から endTime まで並列再開する。

    例: python -m src.main continue-run ./output/karman_re1000/pimpleFoam_cylinder_2d_ogrid -e 200 --np 4
    """
    from pathlib import Path
    from .case_runtime import find_latest_time, read_end_time
    from .runner import OpenFOAMRunner

    case_path = Path(case_dir).resolve()
    if not (case_path / "system" / "controlDict").exists():
        console.print(f"[red]エラー: OpenFOAM ケースが見つかりません: {case_path}[/red]")
        raise typer.Exit(1)

    latest = find_latest_time(case_path)
    current_end = read_end_time(case_path)
    if latest is None:
        console.print("[red]エラー: タイムディレクトリがありません[/red]")
        raise typer.Exit(1)
    if end_time <= latest:
        console.print(
            f"[red]エラー: --end-time ({end_time}) は最新時刻 ({latest}) より大きくしてください[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[bold]計算続行[/bold]  t={latest:g} → endTime={end_time:g}"
        + (f" (現在 endTime={current_end:g})" if current_end else "")
    )

    settings = Settings()
    runner = OpenFOAMRunner(settings)
    result = runner.run_continue_solver(
        str(case_path),
        end_time,
        n_procs=n_procs,
        write_interval=write_interval,
    )
    if not result.success:
        console.print(f"[red]ソルバーが失敗しました (log: {result.log_file})[/red]")
        raise typer.Exit(1)

    if not no_vtk:
        runner.run_foam_to_vtk(str(case_path))

    n_times = len(list(case_path.glob("[0-9]*")))
    console.print(f"[bold green]完了[/bold green]  タイムディレクトリ数: {n_times}")


@app.command()
def reconstruct(
    case_dir: str = typer.Argument(..., help="復元する OpenFOAM ケースディレクトリ"),
    latest_only: bool = typer.Option(
        False, "--latest-only", help="最新タイムステップのみ復元（デフォルト: 全タイムステップ）"
    ),
    vtk: bool = typer.Option(True, "--vtk/--no-vtk", help="復元後に foamToVTK を実行"),
):
    """
    並列計算後の processor* から reconstructPar でタイムステップを復元する。

    例: python -m src.main reconstruct ./output/karman_re1000/pimpleFoam_cylinder_2d_ogrid
    """
    from pathlib import Path
    from .case_runtime import has_processor_dirs, list_time_dirs
    from .runner import OpenFOAMRunner

    case_path = Path(case_dir).resolve()
    if not has_processor_dirs(case_path):
        console.print("[yellow]processor* がありません — スキップ[/yellow]")
    else:
        settings = Settings()
        runner = OpenFOAMRunner(settings)
        result = runner.run_reconstruct_par(str(case_path), latest_only=latest_only)
        if not result.success:
            console.print(f"[red]reconstructPar 失敗 (log: {result.log_file})[/red]")
            raise typer.Exit(1)

    if vtk:
        settings = Settings()
        OpenFOAMRunner(settings).run_foam_to_vtk(str(case_path))

    times = list_time_dirs(case_path)
    console.print(
        f"[bold green]完了[/bold green]  ルートのタイムディレクトリ: {len(times)} "
        f"(t={times[0]:g} … {times[-1]:g})" if times else "[bold green]完了[/bold green]"
    )


@app.command()
def chat(
    workspace: str = typer.Option(
        ...,
        "--workspace",
        "-w",
        help="OpenFOAM ケースのワークスペース（ケースディレクトリ）",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="workspace/.ofagent/session.json から前回セッションを復元",
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        help="編集・実行の確認ゲートを省略（デフォルトは確認あり）",
    ),
):
    """
    【対話型エージェント】ケースディレクトリをワークスペースとしてチャットで操作する。

    例: python -m src.main chat --workspace ./output/my_case
    例: python -m src.main chat -w ./output/my_case --resume
    """
    from pathlib import Path

    from .cli_chat import run_chat

    run_chat(Path(workspace), resume=resume, yolo=yolo)


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

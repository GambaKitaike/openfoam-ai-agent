"""
後処理・可視化モジュール
ParaView 起動・結果サマリー生成を担当
"""
from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .runner import OpenFOAMRunner

console = Console()


class PostProcessor:
    """解析結果の後処理と可視化を担当するクラス。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runner = OpenFOAMRunner(settings)

    def run(self, case_dir: str, open_paraview: bool = True) -> dict:
        """
        後処理パイプラインを実行する。

        手順:
          1. foamToVTK で VTK 形式に変換
          2. .foam ファイルを作成
          3. ParaView を起動

        Args:
            case_dir: ケースディレクトリのパス
            open_paraview: True なら ParaView を自動起動

        Returns:
            dict: 後処理結果の情報
        """
        case_path = Path(case_dir)
        results = {}

        # .foam ファイルを作成（ParaView がケースを認識するためのマーカー）
        foam_file = self._create_foam_file(case_path)
        results["foam_file"] = str(foam_file)
        console.print(f"[green]  ✓ .foam ファイル作成: {foam_file}[/green]")

        # foamToVTK 実行（VTK変換 - 失敗しても続行）
        vtk_result = self.runner.run_foam_to_vtk(case_dir)
        results["vtk_success"] = vtk_result.success
        if vtk_result.success:
            vtk_dir = case_path / "VTK"
            results["vtk_dir"] = str(vtk_dir)
            console.print(f"[green]  ✓ VTK変換完了: {vtk_dir}[/green]")
        else:
            console.print("[yellow]  ⚠ foamToVTK はスキップ (結果データがない可能性があります)[/yellow]")

        # ParaView の起動
        if open_paraview:
            results.update(self._launch_paraview(case_path, foam_file))

        return results

    def _create_foam_file(self, case_path: Path) -> Path:
        """ParaView 用の .foam マーカーファイルを作成する。"""
        foam_file = case_path / f"{case_path.name}.foam"
        foam_file.touch()
        return foam_file

    def _launch_paraview(self, case_path: Path, foam_file: Path) -> dict:
        """ParaView を起動する。"""
        paraview_bin = self._find_paraview()

        if paraview_bin is None:
            console.print(Panel(
                "[yellow]ParaView が見つかりません。\n\n"
                "インストールするには:\n"
                "[cyan]sudo apt install paraview[/cyan]\n\n"
                "または公式サイトからダウンロード:\n"
                "[cyan]https://www.paraview.org/download/[/cyan]\n\n"
                f"インストール後、以下のコマンドで開けます:\n"
                f"[cyan]paraview {foam_file}[/cyan][/yellow]",
                title="[yellow]ParaView が未インストール[/yellow]",
                border_style="yellow",
            ))
            return {"paraview_launched": False, "paraview_cmd": f"paraview {foam_file}"}

        import subprocess
        import os

        # WSL2 環境での OpenMPI / XDG 問題を回避する環境変数
        env = os.environ.copy()
        env.update({
            # OpenMPI が network interface を見つけられない WSL2 問題の回避
            "OMPI_MCA_btl": "tcp,self",
            "OMPI_MCA_btl_tcp_if_include": "lo",
            "OMPI_MCA_opal_warn_on_missing_libcuda": "0",
            # XDG runtime dir がない場合の回避
            "XDG_RUNTIME_DIR": "/tmp/paraview-xdg-runtime",
        })

        # XDG_RUNTIME_DIR を作成
        import stat
        xdg_dir = Path(env["XDG_RUNTIME_DIR"])
        xdg_dir.mkdir(parents=True, exist_ok=True)
        xdg_dir.chmod(stat.S_IRWXU)  # 700 (所有者のみアクセス可)

        cmd = f"{paraview_bin} {foam_file}"
        console.print(f"[green]  ✓ ParaView を起動: {cmd}[/green]")
        proc = subprocess.Popen(cmd, shell=True, env=env)

        # 少し待って起動に失敗していないか確認
        import time
        time.sleep(2)
        if proc.poll() is not None:
            # 即座に終了 → WSLg の問題の可能性
            windows_path = PostProcessor._to_windows_path(case_path)
            console.print(Panel(
                "[yellow]WSL 側の ParaView が正常に起動できませんでした。\n\n"
                "[bold]Windows の ParaView で開いてください:[/bold]\n"
                "  1. paraview.org からWindows版をインストール\n"
                f"  2. エクスプローラーで開く:\n"
                f"     [cyan]{windows_path}[/cyan]\n"
                f"  3. [bold]{foam_file.name}[/bold] を ParaView にドラッグ＆ドロップ[/yellow]",
                title="[yellow]ParaView 起動方法[/yellow]",
                border_style="yellow",
            ))
            return {"paraview_launched": False, "paraview_cmd": cmd}

        return {"paraview_launched": True, "paraview_cmd": cmd}

    def _find_paraview(self) -> str | None:
        """ParaView の実行ファイルを探す。"""
        for name in ["paraview", "paraFoam"]:
            path = shutil.which(name)
            if path:
                return path
        # 一般的なインストールパス
        common_paths = [
            "/usr/bin/paraview",
            "/usr/local/bin/paraview",
            "/opt/paraview/bin/paraview",
        ]
        for p in common_paths:
            if Path(p).exists():
                return p
        return None

    def print_result_summary(self, case_dir: str):
        """解析結果のサマリーを表示する。"""
        case_path = Path(case_dir)
        table = Table(title="解析結果ファイル", show_header=True, header_style="bold cyan")
        table.add_column("ファイル/フォルダ", style="white")
        table.add_column("説明", style="dim")

        entries = [
            (f"log.{p.name[4:]}", "ソルバーログ")
            for p in case_path.glob("log.*")
        ]
        entries += [
            (str(p.name), "結果タイムステップ")
            for p in sorted(case_path.iterdir())
            if p.is_dir() and p.name.replace(".", "").isdigit() and p.name != "0"
        ]
        foam_files = list(case_path.glob("*.foam"))
        if foam_files:
            entries.append((foam_files[0].name, "ParaView 用ファイル"))

        for name, desc in entries:
            table.add_row(name, desc)

        console.print(table)

        foam_file = case_path / f"{case_path.name}.foam"
        windows_path = self._to_windows_path(case_path)

        console.print(Panel(
            f"[bold]Linux (WSL) から開く:[/bold]\n"
            f"  [cyan]paraview {foam_file}[/cyan]\n\n"
            f"[bold]Windows の ParaView から開く (推奨):[/bold]\n"
            f"  エクスプローラーのアドレスバーに入力:\n"
            f"  [cyan]{windows_path}[/cyan]\n"
            f"  → [bold]{case_path.name}.foam[/bold] を Windows の ParaView で開く",
            title="[bold green]ParaView で開く方法[/bold green]",
            border_style="green",
        ))

    @staticmethod
    def _to_windows_path(linux_path: Path) -> str:
        r"""Linux パスを Windows の UNC パス (\\wsl$\...) に変換する。"""
        import subprocess
        try:
            # wslpath コマンドで変換
            result = subprocess.run(
                ["wslpath", "-w", str(linux_path)],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        # フォールバック: 手動変換
        path_str = str(linux_path)
        return f"\\\\wsl$\\Ubuntu{path_str}"

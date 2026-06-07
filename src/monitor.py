"""
ソルバー収束モニタリングモジュール
ログファイルをパースして残差をリアルタイム表示する
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()

# OpenFOAMログから残差を抽出する正規表現
# 例: "smoothSolver:  Solving for Ux, Initial residual = 0.1234, ..."
_RESIDUAL_PATTERN = re.compile(
    r"Solving for (\w+),\s+Initial residual = ([0-9.eE+\-]+)"
)
# 例: "Time = 100"
_TIME_PATTERN = re.compile(r"^Time = ([0-9.eE+\-]+)", re.MULTILINE)

# 収束判定のデフォルト閾値
DEFAULT_CONVERGENCE_THRESHOLD = 1e-4


@dataclass
class ResidualRecord:
    time_step: float
    field: str
    residual: float


@dataclass
class ConvergenceStatus:
    time_step: float = 0.0
    residuals: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    diverged: bool = False
    iterations: int = 0

    def is_converged(self, threshold: float = DEFAULT_CONVERGENCE_THRESHOLD) -> bool:
        if not self.residuals:
            return False
        return all(v < threshold for v in self.residuals.values())

    def is_diverged(self) -> bool:
        return any(
            v > 1e3 or (v != v)  # NaN check
            for v in self.residuals.values()
        )


class SolverMonitor:
    """ソルバーのログファイルを監視して収束状況を追跡するクラス。"""

    def __init__(self, log_file: str, convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD):
        self.log_file = Path(log_file)
        self.convergence_threshold = convergence_threshold
        self.history: list[ResidualRecord] = []
        self._status = ConvergenceStatus()

    def parse_log(self) -> ConvergenceStatus:
        """ログファイル全体をパースして最新の収束状況を返す。"""
        if not self.log_file.exists():
            return self._status

        text = self.log_file.read_text()
        self._parse_text(text)
        return self._status

    def _parse_text(self, text: str):
        """テキストから残差と時刻を抽出して status を更新する。"""
        # 最新のタイムステップを取得
        times = _TIME_PATTERN.findall(text)
        if times:
            self._status.time_step = float(times[-1])
            self._status.iterations = len(times)

        # 最新タイムステップの残差のみ取得（最後のブロック）
        last_time_pos = max(
            (m.start() for m in re.finditer(r"^Time = ", text, re.MULTILINE)),
            default=0,
        )
        last_block = text[last_time_pos:]

        residuals: dict[str, float] = {}
        for match in _RESIDUAL_PATTERN.finditer(last_block):
            field_name = match.group(1)
            residual_val = float(match.group(2))
            residuals[field_name] = residual_val

        if residuals:
            self._status.residuals = residuals

        self._status.converged = self._status.is_converged(self.convergence_threshold)
        self._status.diverged = self._status.is_diverged()

    def watch(self, solver_result_fn, poll_interval: float = 2.0):
        """
        ソルバーの実行を監視しながらリアルタイムで残差を表示する。

        Args:
            solver_result_fn: ソルバーを実行する関数 (RunResult を返す)
            poll_interval: ログを再読み込みする間隔 (秒)
        """
        import threading
        result_holder = [None]

        def run_solver():
            result_holder[0] = solver_result_fn()

        thread = threading.Thread(target=run_solver, daemon=True)
        thread.start()

        with Live(self._make_table(), refresh_per_second=1, console=console) as live:
            while thread.is_alive():
                time.sleep(poll_interval)
                self.parse_log()
                live.update(self._make_table())

        # 最終状態を表示
        self.parse_log()
        console.print(self._make_table())

        return result_holder[0]

    def _make_table(self) -> Panel:
        """Rich のテーブルで残差を表示する。"""
        table = Table(show_header=True, header_style="bold cyan", expand=False)
        table.add_column("フィールド", style="bold white")
        table.add_column("残差", justify="right")
        table.add_column("収束?", justify="center")

        for field_name, residual in self._status.residuals.items():
            converged = residual < self.convergence_threshold
            residual_str = f"{residual:.3e}"
            status_str = "[green]✓[/green]" if converged else "[yellow]…[/yellow]"
            table.add_row(field_name, residual_str, status_str)

        status_text = ""
        if self._status.converged:
            status_text = "[bold green]収束しました！[/bold green]"
        elif self._status.diverged:
            status_text = "[bold red]発散しています！[/bold red]"
        else:
            status_text = f"[white]計算中... (Step {self._status.iterations})[/white]"

        return Panel(
            table,
            title=f"[bold]残差モニタ - Time = {self._status.time_step}[/bold]",
            subtitle=status_text,
            border_style="cyan",
        )

    def summary(self) -> str:
        """収束サマリーを文字列で返す。"""
        lines = [f"最終タイムステップ: {self._status.time_step}"]
        lines.append(f"総イテレーション数: {self._status.iterations}")
        for field_name, residual in self._status.residuals.items():
            lines.append(f"  {field_name}: {residual:.3e}")
        if self._status.converged:
            lines.append("→ 収束しました")
        elif self._status.diverged:
            lines.append("→ 発散しました（設定を見直してください）")
        else:
            lines.append("→ 収束未達（endTime を増やしてください）")
        return "\n".join(lines)

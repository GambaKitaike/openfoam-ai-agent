"""
Agent④ Post-processing Agent
解析結果の物理妥当性チェック、レポート生成、Windows ParaView 案内
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import Settings
from ..models import CaseArtifacts, AnalysisReport, PhysicsCheck

console = Console()

# 物理妥当性チェックの閾値
LAMINAR_RE_THRESHOLD = 2300      # Re < 2300: 層流
TURBULENT_RE_THRESHOLD = 4000    # Re > 4000: 乱流


class PostprocessingAgent:
    """Agent④: 物理妥当性チェック・レポート生成・可視化案内エージェント。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, artifacts: CaseArtifacts) -> AnalysisReport:
        """
        CaseArtifacts を受け取り、後処理・レポート生成を行う。

        Args:
            artifacts: Agent③ が生成した解析成果物

        Returns:
            AnalysisReport
        """
        checks = self._run_physics_checks(artifacts)
        overall_valid = sum(1 for c in checks if c.passed) >= len(checks) * 0.7

        # .foam ファイル作成
        foam_file = self._create_foam_file(artifacts.case_dir)
        artifacts.foam_file = str(foam_file)

        # foamToVTK 実行（失敗してもレポートは続行）
        vtk_dir = self._run_foam_to_vtk(artifacts.case_dir)
        if vtk_dir:
            artifacts.vtk_dir = str(vtk_dir)

        # Windows パス
        windows_path = self._to_windows_path(Path(artifacts.case_dir))

        # レポート生成
        report_text = self._build_report_text(artifacts, checks, windows_path)
        report_file = self._save_report(artifacts.case_dir, report_text)

        report = AnalysisReport(
            artifacts=artifacts,
            physics_checks=checks,
            overall_valid=overall_valid,
            summary_text=report_text,
            report_file=str(report_file),
            windows_paraview_path=windows_path,
        )

        self._print_summary(report)
        return report

    # ──────────────────────────────────────────────────────────────────
    # 物理妥当性チェック
    # ──────────────────────────────────────────────────────────────────

    def _run_physics_checks(self, artifacts: CaseArtifacts) -> list[PhysicsCheck]:
        """各種物理妥当性チェックを実行する。"""
        checks = []
        spec = artifacts.spec

        # ① 収束チェック
        if spec.steady_state:
            checks.append(PhysicsCheck(
                name="残差収束",
                passed=artifacts.converged,
                message="収束しました" if artifacts.converged else "収束未達（endTime 増加を推奨）",
                value=artifacts.final_residuals,
            ))
        else:
            checks.append(PhysicsCheck(
                name="非定常計算完了",
                passed=artifacts.converged,
                message="endTime に到達しました" if artifacts.converged else "計算が途中で停止しました",
                value=artifacts.final_residuals,
            ))

        # ② Re 数と乱流モデルの整合性チェック
        re = spec.re_number
        if re is not None:
            if re < LAMINAR_RE_THRESHOLD:
                correct_model = spec.turbulence_model == "laminar"
                checks.append(PhysicsCheck(
                    name="Re数 vs 乱流モデル",
                    passed=correct_model,
                    message=f"Re={re:.0f} → 層流域。turbulenceModel=laminar が適切" if not correct_model
                            else f"Re={re:.0f} → 層流、laminar モデルが適切に設定されています",
                    value=re,
                ))
            elif re > TURBULENT_RE_THRESHOLD:
                correct_model = spec.turbulence_model != "laminar"
                # 定常 × 高 Re の組み合わせ警告
                steady_high_re = spec.steady_state and re > 10000
                checks.append(PhysicsCheck(
                    name="Re数 vs 乱流モデル",
                    passed=correct_model,
                    message=(f"Re={re:.0f} → 乱流域。乱流モデルが適切に設定されています"
                             + (" ⚠ Re > 10,000 の外部流れは非定常(pimpleFoam)推奨" if steady_high_re else ""))
                            if correct_model else f"Re={re:.0f} → 乱流域なのに laminar モデルが設定されています",
                    value=re,
                ))
            else:
                checks.append(PhysicsCheck(
                    name="Re数 vs 乱流モデル",
                    passed=True,
                    message=f"Re={re:.0f} → 遷移域（層流・乱流どちらでも可）",
                    value=re,
                ))

        # ③ 最終残差の値チェック
        if artifacts.final_residuals:
            max_residual = max(artifacts.final_residuals.values())
            checks.append(PhysicsCheck(
                name="最終残差",
                passed=max_residual < 0.01,
                message=f"最大残差: {max_residual:.2e}" + (" (良好)" if max_residual < 0.01 else " (要注意)"),
                value=max_residual,
            ))

        # ④ blockMesh 成功チェック
        checks.append(PhysicsCheck(
            name="メッシュ生成",
            passed=artifacts.block_mesh_success,
            message="メッシュ生成成功" if artifacts.block_mesh_success else "メッシュ生成失敗",
        ))

        # ⑤ ソルバー実行チェック
        checks.append(PhysicsCheck(
            name="ソルバー実行",
            passed=artifacts.solver_success,
            message="ソルバー正常終了" if artifacts.solver_success else "ソルバー異常終了",
        ))

        # ⑥ 計算領域の速度チェック（結果ファイルがあれば）
        velocity_check = self._check_velocity_range(artifacts)
        if velocity_check:
            checks.append(velocity_check)

        return checks

    def _check_velocity_range(self, artifacts: CaseArtifacts) -> PhysicsCheck | None:
        """速度の値域が物理的に妥当かチェックする（ログから推定）。"""
        log_file = artifacts.log_files.get(artifacts.spec.solver, "")
        if not log_file or not Path(log_file).exists():
            return None
        try:
            import re
            text = Path(log_file).read_text()
            # "Courant Number mean: X max: Y" を探す
            matches = re.findall(r'max\(U\)\s*=\s*([\d.eE+\-]+)', text)
            if matches:
                max_u = float(matches[-1])
                expected = artifacts.spec.inlet_velocity
                ratio = max_u / expected if expected > 0 else 0
                passed = 0.1 < ratio < 20
                return PhysicsCheck(
                    name="速度値域",
                    passed=passed,
                    message=f"最大速度 {max_u:.2f} m/s (流入速度の {ratio:.1f} 倍)" +
                            (" → 物理的に妥当" if passed else " → 異常値の可能性あり"),
                    value=max_u,
                )
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────────
    # ファイル操作
    # ──────────────────────────────────────────────────────────────────

    def _create_foam_file(self, case_dir: str) -> Path:
        """ParaView 用 .foam マーカーファイルを作成する。"""
        case_path = Path(case_dir)
        foam_file = case_path / f"{case_path.name}.foam"
        foam_file.touch()
        return foam_file

    def _run_foam_to_vtk(self, case_dir: str) -> Path | None:
        """foamToVTK を実行してVTKファイルを生成する。"""
        from ..runner import OpenFOAMRunner
        runner = OpenFOAMRunner(self.settings)
        result = runner.run_foam_to_vtk(case_dir)
        vtk_dir = Path(case_dir) / "VTK"
        return vtk_dir if result.success and vtk_dir.exists() else None

    def _save_report(self, case_dir: str, report_text: str) -> Path:
        """Markdown レポートを保存する。"""
        report_path = Path(case_dir) / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        return report_path

    # ──────────────────────────────────────────────────────────────────
    # レポート生成
    # ──────────────────────────────────────────────────────────────────

    def _build_report_text(
        self, artifacts: CaseArtifacts,
        checks: list[PhysicsCheck],
        windows_path: str,
    ) -> str:
        """Markdown 形式のレポートを生成する。"""
        spec = artifacts.spec
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)

        lines = [
            f"# OpenFOAM 解析レポート",
            f"",
            f"**生成日時**: {now}  ",
            f"**解析内容**: {spec.description}  ",
            f"**ケースディレクトリ**: `{artifacts.case_dir}`",
            f"",
            f"---",
            f"",
            f"## 解析仕様",
            f"",
            f"| 項目 | 値 |",
            f"|------|-----|",
            f"| ソルバー | `{spec.solver}` |",
            f"| 解析タイプ | {spec.case_type} |",
            f"| 乱流モデル | {spec.turbulence_model} |",
            f"| 定常/非定常 | {'定常' if spec.steady_state else '非定常'} |",
            f"| 流入速度 | {spec.inlet_velocity} m/s |",
        ]
        if spec.re_number:
            lines.append(f"| Re数 | {spec.re_number:,.0f} |")

        lines += [
            f"",
            f"---",
            f"",
            f"## 物理妥当性チェック ({passed}/{total} 合格)",
            f"",
        ]
        for c in checks:
            icon = "✅" if c.passed else "⚠️"
            lines.append(f"- {icon} **{c.name}**: {c.message}")

        if artifacts.final_residuals:
            lines += [
                f"",
                f"### 最終残差",
                f"",
                f"| フィールド | 残差 |",
                f"|-----------|------|",
            ]
            for field, val in artifacts.final_residuals.items():
                lines.append(f"| {field} | {val:.3e} |")

        lines += [
            f"",
            f"---",
            f"",
            f"## ParaView での可視化",
            f"",
            f"### Windows の ParaView から開く（推奨）",
            f"",
            f"エクスプローラーのアドレスバーに入力:",
            f"```",
            f"{windows_path}",
            f"```",
            f"`{Path(artifacts.case_dir).name}.foam` を Windows の ParaView で開いてください。",
            f"",
            f"### WSL から開く",
            f"```bash",
            f"paraview {artifacts.foam_file}",
            f"```",
        ]
        if spec.defaults_applied:
            lines += [
                f"",
                f"---",
                f"",
                f"## 自動補完されたパラメータ",
                f"",
                f"以下のパラメータはデフォルト値が適用されました:",
            ]
            for d in spec.defaults_applied:
                lines.append(f"- {d}")

        return "\n".join(lines)

    def _print_summary(self, report: AnalysisReport):
        """後処理サマリーをターミナルに表示する。"""
        artifacts = report.artifacts

        # 物理妥当性テーブル
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("チェック項目", style="white")
        table.add_column("結果", justify="center")
        table.add_column("詳細", style="dim")
        for c in report.physics_checks:
            icon = "[green]✓[/green]" if c.passed else "[yellow]⚠[/yellow]"
            table.add_row(c.name, icon, c.message)
        console.print(table)

        overall = "[bold green]物理的に妥当[/bold green]" if report.overall_valid else "[bold yellow]要確認あり[/bold yellow]"
        console.print(Panel(
            f"総合判定: {overall}\n\n"
            f"[bold]Windows ParaView で開く:[/bold]\n"
            f"  エクスプローラー: [cyan]{report.windows_paraview_path}[/cyan]\n"
            f"  → [bold]{Path(artifacts.case_dir).name}.foam[/bold] を開く\n\n"
            f"[bold]レポート:[/bold] [cyan]{report.report_file}[/cyan]",
            title="[bold green]後処理完了[/bold green]",
            border_style="green",
        ))

    @staticmethod
    def _to_windows_path(linux_path: Path) -> str:
        r"""Linux パスを Windows の UNC パス (\\wsl.localhost\...) に変換する。"""
        try:
            result = subprocess.run(
                ["wslpath", "-w", str(linux_path)],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return rf"\\wsl.localhost\Ubuntu-24.04{linux_path}"

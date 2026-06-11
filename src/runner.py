"""
OpenFOAMコマンド実行モジュール
blockMesh・ソルバー・foamToVTK などを subprocess で実行する
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from rich.console import Console

from .config import Settings
from .case_runtime import has_processor_dirs, remove_processor_dirs

console = Console()

# OpenFOAM 環境を読み込むシェルスクリプトのパス
_OF_BASHRC_CANDIDATES = [
    "/usr/lib/openfoam/openfoam2512/etc/bashrc",
    "/opt/openfoam2512/etc/bashrc",
    "/opt/openfoam11/etc/bashrc",
]


def _find_of_bashrc(settings: Settings) -> str | None:
    """OpenFOAMのetc/bashrcを探す。"""
    # 設定で指定されたパスを優先
    candidate = Path(settings.openfoam_root) / "etc" / "bashrc"
    if candidate.exists():
        return str(candidate)
    for path in _OF_BASHRC_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _of_command(cmd: str, settings: Settings) -> str:
    """OpenFOAM環境をsourceしてからコマンドを実行するシェルコマンド文字列を返す。"""
    bashrc = _find_of_bashrc(settings)
    if bashrc:
        return f'bash -c "source {bashrc} && {cmd}"'
    return f'bash -c "{cmd}"'


class RunResult:
    def __init__(self, command: str, returncode: int, stdout: str, stderr: str, log_file: str | None = None):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.log_file = log_file

    @property
    def success(self) -> bool:
        return self.returncode == 0


class OpenFOAMRunner:
    """OpenFOAMコマンドを実行するクラス。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run_block_mesh(self, case_dir: str) -> RunResult:
        """
        blockMesh を実行してメッシュを生成する。
        """
        log_path = Path(case_dir) / "log.blockMesh"
        # pipefail で blockMesh の終了コードを正しく取得する
        cmd_str = _of_command(
            f"set -o pipefail; blockMesh -case {case_dir} 2>&1 | tee {log_path}",
            self.settings
        )

        console.print("[cyan]  → blockMesh を実行中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        # ログに FOAM FATAL ERROR が含まれていても失敗扱いにする
        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode if (result.returncode != 0 or foam_error) else 0
        if foam_error and result.returncode == 0:
            returncode = 1

        return RunResult(
            command="blockMesh",
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_check_mesh(self, case_dir: str) -> RunResult:
        """
        checkMesh を実行してメッシュの品質を確認する。
        """
        log_path = Path(case_dir) / "log.checkMesh"
        cmd_str = _of_command(f"checkMesh -case {case_dir} 2>&1 | tee {log_path}", self.settings)

        console.print("[cyan]  → checkMesh を実行中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        return RunResult(
            command="checkMesh",
            returncode=result.returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_surface_feature_extract(self, case_dir: str) -> RunResult:
        """
        surfaceFeatureExtract を実行してSTLのフィーチャーエッジを抽出する。
        snappyHexMesh の explicitFeatureSnap に必要。
        """
        log_path = Path(case_dir) / "log.surfaceFeatureExtract"
        cmd_str = _of_command(
            f"set -o pipefail; surfaceFeatureExtract -case {case_dir} 2>&1 | tee {log_path}",
            self.settings,
        )

        console.print("[cyan]  → surfaceFeatureExtract を実行中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode
        if foam_error and returncode == 0:
            returncode = 1

        return RunResult(
            command="surfaceFeatureExtract",
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_snappy_hex_mesh(self, case_dir: str) -> RunResult:
        """
        snappyHexMesh を実行して STL 形状に適合したメッシュを生成する。
        """
        log_path = Path(case_dir) / "log.snappyHexMesh"
        cmd_str = _of_command(
            f"set -o pipefail; snappyHexMesh -overwrite -case {case_dir} 2>&1 | tee {log_path}",
            self.settings,
        )

        console.print("[cyan]  → snappyHexMesh を実行中... (数分かかる場合があります)[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=600)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode
        if foam_error and returncode == 0:
            returncode = 1

        return RunResult(
            command="snappyHexMesh",
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_potential_foam(self, case_dir: str) -> RunResult:
        """potentialFoam で速度場を初期化する（pimpleFoam の前処理として使用）。"""
        log_path = Path(case_dir) / "log.potentialFoam"
        cmd = f"set -o pipefail; potentialFoam -case {case_dir} 2>&1 | tee {log_path}"
        cmd_str = _of_command(cmd, self.settings)
        console.print(f"[cyan]  → potentialFoam で速度場を初期化中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout
        success = result.returncode == 0 and "FOAM FATAL" not in stdout
        return RunResult(
            command="potentialFoam",
            returncode=result.returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_set_fields(self, case_dir: str) -> RunResult:
        """setFields で初期場に摂動を与える（カルマン渦用）。"""
        log_path = Path(case_dir) / "log.setFields"
        cmd = f"set -o pipefail; setFields -case {case_dir} 2>&1 | tee {log_path}"
        cmd_str = _of_command(cmd, self.settings)
        console.print("[cyan]  → setFields で後流摂動を付与中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout
        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode if not foam_error else 1
        return RunResult(
            command="setFields",
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def _ensure_decompose_par_dict(self, case_dir: str, n_procs: int) -> None:
        from .case_builder.builders import build_decompose_par_dict

        path = Path(case_dir) / "system" / "decomposeParDict"
        path.write_text(build_decompose_par_dict(n_procs))

    def run_solver(
        self,
        case_dir: str,
        solver: str,
        parallel: bool = False,
        n_procs: int = 4,
        *,
        continue_run: bool = False,
        reconstruct_all: bool = True,
    ) -> RunResult:
        """
        OpenFOAMソルバーを実行する。
        parallel=True のとき decomposePar → mpirun → reconstructPar を行う。
        continue_run=True のとき decomposePar -latestTime で最新状態から再開する。
        """
        log_path = Path(case_dir) / f"log.{solver}"

        if parallel:
            self._ensure_decompose_par_dict(case_dir, n_procs)
            if continue_run:
                console.print(
                    f"[cyan]  → {solver} を {n_procs} 並列で再開 (latestTime → mpirun)[/cyan]"
                )
                remove_processor_dirs(case_dir)
                decompose_cmd = f"decomposePar -case {case_dir} -latestTime -force"
            else:
                console.print(
                    f"[cyan]  → {solver} を {n_procs} 並列で実行 (mpirun)[/cyan]"
                )
                remove_processor_dirs(case_dir)
                decompose_cmd = f"decomposePar -case {case_dir} -time '0' -force"

            recon_flag = "" if reconstruct_all else "-latestTime"
            recon_log = "log.reconstructPar" if reconstruct_all else "log.reconstructPar.latestTime"
            cmd = (
                f"set -o pipefail; "
                f"{decompose_cmd} > {case_dir}/log.decomposePar 2>&1 && "
                f"mpirun -np {n_procs} {solver} -case {case_dir} -parallel 2>&1 | tee {log_path} && "
                f"reconstructPar -case {case_dir} {recon_flag} > {case_dir}/{recon_log} 2>&1"
            )
        else:
            console.print(f"[cyan]  → {solver} を実行中 (ログ: {log_path})[/cyan]")
            cmd = f"set -o pipefail; {solver} -case {case_dir} 2>&1 | tee {log_path}"

        cmd_str = _of_command(cmd, self.settings)
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode
        if foam_error and returncode == 0:
            returncode = 1

        return RunResult(
            command=solver,
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_reconstruct_par(self, case_dir: str, *, latest_only: bool = False) -> RunResult:
        """processor* からタイムステップを復元する。"""
        flag = "-latestTime" if latest_only else ""
        log_name = "log.reconstructPar.latestTime" if latest_only else "log.reconstructPar"
        log_path = Path(case_dir) / log_name
        label = "latestTime のみ" if latest_only else "全タイムステップ"
        console.print(f"[cyan]  → reconstructPar ({label})...[/cyan]")
        cmd_str = _of_command(
            f"set -o pipefail; reconstructPar -case {case_dir} {flag} 2>&1 | tee {log_path}",
            self.settings,
        )
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout
        foam_error = "FOAM FATAL ERROR" in stdout or "FOAM exiting" in stdout
        returncode = result.returncode if not foam_error else 1
        return RunResult(
            command="reconstructPar",
            returncode=returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def ensure_reconstructed_for_viz(self, case_dir: str) -> None:
        """processor* が残っていれば全タイムステップを復元する（ParaView 用）。"""
        if has_processor_dirs(case_dir):
            console.print(
                "[cyan]  → 並列結果を全タイムステップ復元 (reconstructPar)...[/cyan]"
            )
            self.run_reconstruct_par(case_dir, latest_only=False)

    def run_continue_solver(
        self,
        case_dir: str,
        end_time: float,
        *,
        n_procs: int = 4,
        write_interval: float | None = None,
        reconstruct_all: bool = True,
    ) -> RunResult:
        """既存ケースを latestTime から end_time まで並列再開する。"""
        from .case_runtime import patch_control_dict_for_continue, read_solver_name

        solver = read_solver_name(case_dir)
        if not solver:
            raise ValueError(f"controlDict からソルバー名を読み取れません: {case_dir}")

        latest = patch_control_dict_for_continue(
            case_dir, end_time, write_interval=write_interval
        )
        console.print(
            f"[cyan]  → 計算続行: t={latest:g} → endTime={end_time:g} ({solver})[/cyan]"
        )
        return self.run_solver(
            case_dir,
            solver,
            parallel=True,
            n_procs=n_procs,
            continue_run=True,
            reconstruct_all=reconstruct_all,
        )

    def run_foam_to_vtk(self, case_dir: str) -> RunResult:
        """
        foamToVTK を実行してParaView用のVTKファイルを生成する。
        """
        log_path = Path(case_dir) / "log.foamToVTK"
        cmd_str = _of_command(f"foamToVTK -case {case_dir} 2>&1 | tee {log_path}", self.settings)

        console.print("[cyan]  → foamToVTK を実行中...[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        stdout = log_path.read_text() if log_path.exists() else result.stdout

        return RunResult(
            command="foamToVTK",
            returncode=result.returncode,
            stdout=stdout,
            stderr=result.stderr,
            log_file=str(log_path),
        )

    def run_parafoam(self, case_dir: str) -> RunResult:
        """
        paraFoam を起動してParaViewでケースを開く。
        """
        # paraFoam用の .foam ファイルを作成
        case_path = Path(case_dir)
        foam_file = case_path / f"{case_path.name}.foam"
        foam_file.touch()

        # paraviewのパスを探す
        paraview_cmd = shutil.which("paraview") or shutil.which("paraFoam")

        if paraview_cmd is None:
            return RunResult(
                command="paraview",
                returncode=1,
                stdout="",
                stderr="ParaView が見つかりません。インストールしてください: sudo apt install paraview",
            )

        cmd_str = f"{paraview_cmd} {foam_file} &"
        console.print(f"[cyan]  → ParaView を起動中: {foam_file}[/cyan]")
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)

        return RunResult(
            command="paraview",
            returncode=result.returncode,
            stdout=f"ParaView を起動しました。ファイル: {foam_file}",
            stderr=result.stderr,
        )

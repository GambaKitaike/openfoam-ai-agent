"""OpenFOAM ケースのランタイム操作（時刻ディレクトリ・controlDict 更新）。"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


def list_time_dirs(case_dir: str | Path) -> list[float]:
    """ケースルートの数値タイムディレクトリ名を昇順で返す。"""
    case_path = Path(case_dir)
    times: list[float] = []
    for p in case_path.iterdir():
        if not p.is_dir() or p.name.startswith("processor"):
            continue
        try:
            times.append(float(p.name))
        except ValueError:
            continue
    return sorted(times)


def find_latest_time(case_dir: str | Path) -> float | None:
    times = list_time_dirs(case_dir)
    return times[-1] if times else None


def has_processor_dirs(case_dir: str | Path) -> bool:
    return any(Path(case_dir).glob("processor*"))


def read_solver_name(case_dir: str | Path) -> str | None:
    text = (Path(case_dir) / "system" / "controlDict").read_text(errors="ignore")
    m = re.search(r"application\s+(\w+)\s*;", text)
    return m.group(1) if m else None


def read_end_time(case_dir: str | Path) -> float | None:
    text = (Path(case_dir) / "system" / "controlDict").read_text(errors="ignore")
    m = re.search(r"endTime\s+([\d.eE+-]+)\s*;", text)
    return float(m.group(1)) if m else None


def patch_control_dict_for_continue(
    case_dir: str | Path,
    end_time: float,
    *,
    write_interval: float | None = None,
) -> float:
    """startFrom latestTime と endTime を設定。直前 latest time を返す。"""
    case_path = Path(case_dir)
    latest = find_latest_time(case_path)
    if latest is None:
        raise ValueError(f"タイムディレクトリが見つかりません: {case_path}")

    path = case_path / "system" / "controlDict"
    text = path.read_text()
    text = re.sub(r"startFrom\s+\S+;", "startFrom       latestTime;", text, count=1)
    text = re.sub(r"endTime\s+[\d.eE+-]+;", f"endTime         {end_time:g};", text, count=1)
    if write_interval is not None:
        text = re.sub(
            r"writeInterval\s+[\d.eE+-]+;",
            f"writeInterval   {write_interval:g};",
            text,
            count=1,
        )
    path.write_text(text)
    return latest


def remove_processor_dirs(case_dir: str | Path) -> None:
    case_path = Path(case_dir)
    for p in case_path.glob("processor*"):
        if p.is_dir():
            shutil.rmtree(p)

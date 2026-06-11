"""SessionState の永続化（DESIGN.md §3.1, §3.2）。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import SimulationSpec

Message = dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunRecord:
    command: str
    log_path: Path
    exit_code: int
    started_at: datetime
    finished_at: datetime
    summary: str


@dataclass
class SessionState:
    workspace: Path
    spec: SimulationSpec | None = None
    history: list[Message] = field(default_factory=list)
    run_records: list[RunRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


def _session_path(workspace: Path) -> Path:
    return workspace.resolve() / ".ofagent" / "session.json"


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _deserialize_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _serialize_spec(spec: SimulationSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return asdict(spec)


def _deserialize_spec(data: dict[str, Any] | None) -> SimulationSpec | None:
    if data is None:
        return None
    valid_keys = {item.name for item in fields(SimulationSpec)}
    filtered = {key: value for key, value in data.items() if key in valid_keys}
    return SimulationSpec(**filtered)


def _serialize_run_record(record: RunRecord) -> dict[str, Any]:
    return {
        "command": record.command,
        "log_path": record.log_path.as_posix(),
        "exit_code": record.exit_code,
        "started_at": _serialize_datetime(record.started_at),
        "finished_at": _serialize_datetime(record.finished_at),
        "summary": record.summary,
    }


def _deserialize_run_record(data: dict[str, Any]) -> RunRecord:
    return RunRecord(
        command=data["command"],
        log_path=Path(data["log_path"]),
        exit_code=int(data["exit_code"]),
        started_at=_deserialize_datetime(data["started_at"]),
        finished_at=_deserialize_datetime(data["finished_at"]),
        summary=data["summary"],
    )


def _serialize_state(state: SessionState) -> dict[str, Any]:
    return {
        "workspace": state.workspace.resolve().as_posix(),
        "spec": _serialize_spec(state.spec),
        "history": state.history,
        "run_records": [_serialize_run_record(record) for record in state.run_records],
        "created_at": _serialize_datetime(state.created_at),
        "updated_at": _serialize_datetime(state.updated_at),
    }


def _deserialize_state(data: dict[str, Any]) -> SessionState:
    return SessionState(
        workspace=Path(data["workspace"]),
        spec=_deserialize_spec(data.get("spec")),
        history=list(data.get("history", [])),
        run_records=[_deserialize_run_record(item) for item in data.get("run_records", [])],
        created_at=_deserialize_datetime(data["created_at"]),
        updated_at=_deserialize_datetime(data["updated_at"]),
    )


def save(state: SessionState) -> None:
    """workspace/.ofagent/session.json に SessionState を保存する。"""
    state.updated_at = _utc_now()
    session_file = _session_path(state.workspace)
    session_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_state(state)
    session_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(workspace: Path) -> SessionState:
    """workspace/.ofagent/session.json から SessionState を復元する。"""
    session_file = _session_path(workspace)
    if not session_file.exists():
        return SessionState(workspace=workspace.resolve())
    data = json.loads(session_file.read_text(encoding="utf-8"))
    state = _deserialize_state(data)
    state.workspace = workspace.resolve()
    return state

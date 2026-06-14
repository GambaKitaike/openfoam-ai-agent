"""SessionState の永続化テスト。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.session import SessionState, load, save, sanitize_for_json, sanitize_utf8_text


class TestSanitizeUtf8:
    def test_replaces_lone_surrogate(self) -> None:
        assert sanitize_utf8_text("bad\udce3text") == "bad?text"

    def test_sanitize_for_json_recurses(self) -> None:
        payload = {
            "history": [{"role": "tool", "content": "x\udce3y"}],
            "nested": ["a\udce3b"],
        }
        sanitized = sanitize_for_json(payload)
        assert sanitized["history"][0]["content"] == "x?y"
        assert sanitized["nested"][0] == "a?b"


class TestSessionSave:
    def test_save_with_surrogate_in_history(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)
        state.history = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "log line with bad byte\udce3here",
            }
        ]

        save(state)

        session_file = tmp_path / ".ofagent" / "session.json"
        assert session_file.is_file()
        raw = session_file.read_bytes()
        assert b"\xed\xb3\xa3" not in raw  # UTF-8 for lone surrogate must not appear

        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert "\udce3" not in data["history"][0]["content"]
        assert "?" in data["history"][0]["content"]

        restored = load(tmp_path)
        assert restored.history[0]["content"] == data["history"][0]["content"]

    def test_save_with_surrogate_in_run_record_summary(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        from src.agent.session import RunRecord

        now = datetime.now(timezone.utc)
        state = SessionState(workspace=tmp_path)
        state.run_records = [
            RunRecord(
                command="blockMesh",
                log_path=Path("log.blockMesh"),
                exit_code=1,
                started_at=now,
                finished_at=now,
                summary="failed\udce3summary",
            )
        ]

        save(state)

        restored = load(tmp_path)
        assert "\udce3" not in restored.run_records[0].summary
        assert restored.run_records[0].summary == "failed?summary"

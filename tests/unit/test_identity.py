"""Tests for the machine identity module."""
import json
import os
import uuid

import pytest

from env_doctor.identity import (
    get_machine_id,
    get_machine_info,
    should_report,
    mark_reported,
    _hash_report,
    save_report_config,
    load_report_config,
    remove_report_config,
)


@pytest.fixture(autouse=True)
def isolate_env_doctor_dir(tmp_path, monkeypatch):
    """Redirect ~/.env-doctor to a temp directory for all tests."""
    import env_doctor.identity as mod

    monkeypatch.setattr(mod, "_ENV_DOCTOR_DIR", tmp_path)
    monkeypatch.delenv("ENV_DOCTOR_MACHINE_ID", raising=False)


class TestGetMachineId:
    def test_generates_uuid(self):
        mid = get_machine_id()
        uuid.UUID(mid)  # Should not raise

    def test_persists_across_calls(self):
        first = get_machine_id()
        second = get_machine_id()
        assert first == second

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENV_DOCTOR_MACHINE_ID", "custom-id-123")
        assert get_machine_id() == "custom-id-123"


class TestGetMachineInfo:
    def test_returns_required_fields(self):
        info = get_machine_info()
        assert "machine_id" in info
        assert "hostname" in info
        assert "platform" in info
        assert "python_version" in info
        assert "reported_at" in info


class TestChangeDetection:
    SAMPLE_REPORT = {
        "status": "pass",
        "summary": {"driver": "found", "cuda": "found", "issues_count": 0},
        "checks": {"driver": {"version": "535.0"}},
    }

    def test_first_report_always_sends(self):
        assert should_report(self.SAMPLE_REPORT) is True

    def test_same_report_skips(self):
        mark_reported(self.SAMPLE_REPORT, "http://localhost:8765")
        assert should_report(self.SAMPLE_REPORT, heartbeat_seconds=9999) is False

    def test_changed_report_sends(self):
        mark_reported(self.SAMPLE_REPORT, "http://localhost:8765")
        changed = {**self.SAMPLE_REPORT, "status": "fail"}
        assert should_report(changed) is True

    def test_heartbeat_triggers_after_timeout(self):
        mark_reported(self.SAMPLE_REPORT, "http://localhost:8765")
        # With heartbeat_seconds=0, heartbeat is always due
        assert should_report(self.SAMPLE_REPORT, heartbeat_seconds=0) is True

    def test_hash_deterministic(self):
        h1 = _hash_report(self.SAMPLE_REPORT)
        h2 = _hash_report(self.SAMPLE_REPORT)
        assert h1 == h2

    def test_hash_changes_with_content(self):
        h1 = _hash_report(self.SAMPLE_REPORT)
        h2 = _hash_report({**self.SAMPLE_REPORT, "status": "fail"})
        assert h1 != h2


class TestReportConfig:
    def test_save_and_load(self):
        save_report_config("http://localhost:8765", "2m", "30m")
        config = load_report_config()
        assert config is not None
        assert config["url"] == "http://localhost:8765"
        assert config["interval"] == "2m"

    def test_remove(self):
        save_report_config("http://localhost:8765", "2m", "30m")
        remove_report_config()
        assert load_report_config() is None

"""
Machine identity and reporting state management.

Provides stable machine identification and smart change-detection
for the --report-to flag and dashboard reporting.
"""
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_ENV_DOCTOR_DIR = Path.home() / ".env-doctor"


def _ensure_dir() -> Path:
    """Ensure ~/.env-doctor/ exists."""
    _ENV_DOCTOR_DIR.mkdir(exist_ok=True)
    return _ENV_DOCTOR_DIR


def get_machine_id() -> str:
    """Return stable UUID from ~/.env-doctor/machine-id, creating if needed.

    Can be overridden via ENV_DOCTOR_MACHINE_ID environment variable.
    """
    override = os.environ.get("ENV_DOCTOR_MACHINE_ID")
    if override:
        return override

    id_file = _ensure_dir() / "machine-id"
    if id_file.exists():
        return id_file.read_text().strip()

    machine_id = str(uuid.uuid4())
    id_file.write_text(machine_id)
    return machine_id


def get_machine_info() -> dict:
    """Return machine identity envelope for report payloads."""
    return {
        "machine_id": get_machine_id(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Smart reporting: change detection + heartbeat
# ---------------------------------------------------------------------------

_STATE_FILE = "report-state.json"
_DEFAULT_HEARTBEAT_SECONDS = 30 * 60  # 30 minutes


def _hash_report(report: dict) -> str:
    """Deterministic hash of report content (excluding volatile fields)."""
    # Hash status + checks only (ignore timestamps, machine info)
    hashable = {
        "status": report.get("status"),
        "summary": report.get("summary"),
        "checks": report.get("checks"),
    }
    raw = json.dumps(hashable, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_report_state() -> dict:
    """Load ~/.env-doctor/report-state.json or return empty dict."""
    state_path = _ensure_dir() / _STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_report_state(state: dict) -> None:
    """Persist report state to disk."""
    state_path = _ensure_dir() / _STATE_FILE
    state_path.write_text(json.dumps(state, indent=2))


def should_report(report: dict, heartbeat_seconds: int = _DEFAULT_HEARTBEAT_SECONDS) -> bool:
    """Return True if the report should be sent (changed or heartbeat due)."""
    current_hash = _hash_report(report)
    state = _load_report_state()

    previous_hash = state.get("hash")
    if current_hash != previous_hash:
        return True

    # Check heartbeat
    last_heartbeat = state.get("last_heartbeat")
    if not last_heartbeat:
        return True

    try:
        last_dt = datetime.fromisoformat(last_heartbeat)
        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
        return elapsed >= heartbeat_seconds
    except (ValueError, TypeError):
        return True


def mark_reported(report: dict, dashboard_url: str, is_heartbeat: bool = False) -> None:
    """Update report state after a successful POST."""
    now = datetime.now(timezone.utc).isoformat()
    state = _load_report_state()
    state["hash"] = _hash_report(report)
    state["last_reported"] = now
    state["last_heartbeat"] = now
    state["dashboard_url"] = dashboard_url
    if is_heartbeat:
        state["last_heartbeat_only"] = now
    _save_report_state(state)


# ---------------------------------------------------------------------------
# Report config (for report install/uninstall/status)
# ---------------------------------------------------------------------------

_CONFIG_FILE = "report-config.json"


def load_report_config() -> Optional[dict]:
    """Load ~/.env-doctor/report-config.json or return None."""
    config_path = _ensure_dir() / _CONFIG_FILE
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_report_config(url: str, interval: str, heartbeat: str,
                       token: Optional[str] = None) -> None:
    """Persist report config. ``token`` is optional and stored only when given."""
    config_path = _ensure_dir() / _CONFIG_FILE
    payload = {
        "url": url,
        "interval": interval,
        "heartbeat": heartbeat,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    if token:
        payload["token"] = token
    config_path.write_text(json.dumps(payload, indent=2))


def get_report_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the API token used by the host CLI when posting reports.

    Precedence: ``--token`` flag > ``ENV_DOCTOR_API_TOKEN`` env var >
    ``token`` field of ``~/.env-doctor/report-config.json``.
    """
    if cli_token:
        return cli_token.strip() or None
    env_tok = os.environ.get("ENV_DOCTOR_API_TOKEN")
    if env_tok:
        return env_tok.strip() or None
    config = load_report_config() or {}
    cfg_tok = config.get("token")
    if cfg_tok:
        return str(cfg_tok).strip() or None
    return None


def remove_report_config() -> None:
    """Remove report config file."""
    config_path = _ensure_dir() / _CONFIG_FILE
    if config_path.exists():
        config_path.unlink()
    state_path = _ensure_dir() / _STATE_FILE
    if state_path.exists():
        state_path.unlink()

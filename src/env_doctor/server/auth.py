"""Shared-token authentication for the dashboard API.

The dashboard accepts a single bearer token shared between the browser UI
and any host CLIs that POST reports. Token resolution order:

1. ``ENV_DOCTOR_API_TOKEN`` environment variable.
2. ``~/.env-doctor/api-token`` file (auto-generated on first dashboard run).

Set ``ENV_DOCTOR_DISABLE_AUTH=1`` to opt out (intended for local dev only).
"""
import hmac
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, status

_TOKEN_FILE = Path.home() / ".env-doctor" / "api-token"
_DISABLE_FLAG = "ENV_DOCTOR_DISABLE_AUTH"
_ENV_VAR = "ENV_DOCTOR_API_TOKEN"

# Resolved at startup by load_or_create_token() and read by require_token().
_active_token: Optional[str] = None
_auth_disabled: bool = False


def _read_token_file() -> Optional[str]:
    if not _TOKEN_FILE.exists():
        return None
    try:
        value = _TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    return value or None


def _write_token_file(token: str) -> None:
    _TOKEN_FILE.parent.mkdir(exist_ok=True)
    _TOKEN_FILE.write_text(token)
    # Best-effort owner-only permissions; chmod is a no-op on Windows but the
    # parent directory inherits user-only ACLs so the file is still protected.
    try:
        _TOKEN_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def load_or_create_token() -> Optional[str]:
    """Resolve the API token, generating one on first run.

    Returns the active token, or ``None`` when auth is disabled. Side effect:
    sets module-level state used by :func:`require_token`. Prints the token
    to stderr when it is freshly generated so the operator can copy it into
    the browser and host configs.
    """
    global _active_token, _auth_disabled

    if os.environ.get(_DISABLE_FLAG) == "1":
        _auth_disabled = True
        _active_token = None
        print(
            f"⚠️  {_DISABLE_FLAG}=1 — dashboard API is unauthenticated. "
            "Do not expose this port to untrusted networks.",
            file=sys.stderr,
        )
        return None

    _auth_disabled = False

    env_token = os.environ.get(_ENV_VAR)
    if env_token:
        _active_token = env_token.strip()
        return _active_token

    file_token = _read_token_file()
    if file_token:
        _active_token = file_token
        return _active_token

    new_token = secrets.token_urlsafe(32)
    _write_token_file(new_token)
    _active_token = new_token
    print(
        "🔐 Generated new API token at " f"{_TOKEN_FILE}\n"
        f"   Token: {new_token}\n"
        "   Paste it into the dashboard login screen and pass it to host CLIs "
        "via `env-doctor report install --token <token>`.",
        file=sys.stderr,
    )
    return new_token


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency that enforces ``Authorization: Bearer <token>``."""
    if _auth_disabled:
        return

    if _active_token is None:
        # Auth was never initialized — refuse rather than fail open.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API auth not initialized",
        )

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    presented = authorization.split(None, 1)[1].strip()
    if not hmac.compare_digest(presented, _active_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_active_token() -> Optional[str]:
    """Return the resolved token (or ``None`` when auth is disabled)."""
    return _active_token


def token_file_path() -> Path:
    return _TOKEN_FILE

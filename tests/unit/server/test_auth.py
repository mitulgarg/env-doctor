"""Tests for the shared-token auth middleware."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")


VALID_TOKEN = "test-token-abc123"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Test client with auth enabled and a known token."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    import env_doctor.server.database as db_mod
    import env_doctor.server.auth as auth_mod
    from env_doctor.server.database import Base, get_session

    test_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    test_session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_session():
        async with test_session_factory() as session:
            yield session

    async def override_init_db():
        from env_doctor.server import models  # noqa: F401
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(db_mod, "init_db", override_init_db)

    # Pin the token explicitly via the env var path. Redirect the token file
    # to a tmp path so we never touch the user's real ~/.env-doctor/api-token.
    monkeypatch.setenv("ENV_DOCTOR_API_TOKEN", VALID_TOKEN)
    monkeypatch.delenv("ENV_DOCTOR_DISABLE_AUTH", raising=False)
    monkeypatch.setattr(auth_mod, "_TOKEN_FILE", tmp_path / "api-token")
    auth_mod.load_or_create_token()

    from env_doctor.server.app import app
    app.dependency_overrides[get_session] = override_get_session

    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    # Reset module-level auth state so other test files start clean.
    auth_mod._active_token = None
    auth_mod._auth_disabled = False


def test_missing_authorization_returns_401(client):
    resp = client.get("/api/machines")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_wrong_token_returns_401(client):
    resp = client.get(
        "/api/machines",
        headers={"Authorization": "Bearer not-the-right-token"},
    )
    assert resp.status_code == 401


def test_valid_token_passes(client):
    resp = client.get(
        "/api/machines",
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


def test_non_bearer_scheme_returns_401(client):
    # "Token <value>" is rejected — we only accept the Bearer scheme.
    resp = client.get(
        "/api/machines",
        headers={"Authorization": f"Token {VALID_TOKEN}"},
    )
    assert resp.status_code == 401


def test_disable_flag_makes_routes_public(monkeypatch, tmp_path):
    """Setting ENV_DOCTOR_DISABLE_AUTH=1 must short-circuit require_token."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    import env_doctor.server.database as db_mod
    import env_doctor.server.auth as auth_mod
    from env_doctor.server.database import Base, get_session

    test_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    test_session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_session():
        async with test_session_factory() as session:
            yield session

    async def override_init_db():
        from env_doctor.server import models  # noqa: F401
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(db_mod, "init_db", override_init_db)
    monkeypatch.setenv("ENV_DOCTOR_DISABLE_AUTH", "1")
    monkeypatch.delenv("ENV_DOCTOR_API_TOKEN", raising=False)
    monkeypatch.setattr(auth_mod, "_TOKEN_FILE", tmp_path / "api-token")
    auth_mod.load_or_create_token()

    from env_doctor.server.app import app
    app.dependency_overrides[get_session] = override_get_session

    from fastapi.testclient import TestClient
    try:
        with TestClient(app) as c:
            resp = c.get("/api/machines")
            assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()
        auth_mod._active_token = None
        auth_mod._auth_disabled = False

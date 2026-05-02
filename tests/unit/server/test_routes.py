"""Tests for the dashboard API routes using FastAPI TestClient."""
import pytest

# Skip all tests if dashboard dependencies aren't installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")


SAMPLE_REPORT = {
    "machine": {
        "machine_id": "test-uuid-001",
        "hostname": "gpu-node-01",
        "platform": "Linux",
        "platform_release": "5.15.0",
        "python_version": "3.11.5",
        "reported_at": "2025-01-01T00:00:00+00:00",
    },
    "status": "pass",
    "timestamp": "2025-01-01T00:00:00",
    "summary": {
        "driver": "found",
        "cuda": "found",
        "cudnn": "not_found",
        "issues_count": 1,
    },
    "checks": {
        "driver": {
            "component": "nvidia_driver",
            "status": "success",
            "detected": True,
            "version": "535.129.03",
            "path": None,
            "metadata": {"primary_gpu_name": "NVIDIA A100-SXM4-80GB"},
            "issues": [],
            "recommendations": [],
        },
        "cuda": {
            "component": "cuda_toolkit",
            "status": "success",
            "detected": True,
            "version": "12.2",
            "path": "/usr/local/cuda",
            "metadata": {},
            "issues": [],
            "recommendations": [],
        },
        "libraries": {
            "torch": {
                "component": "torch",
                "status": "success",
                "detected": True,
                "version": "2.1.0",
                "path": None,
                "metadata": {"cuda_version": "12.1"},
                "issues": [],
                "recommendations": [],
            },
        },
        "python_compat": {
            "component": "python_compat",
            "status": "success",
            "detected": True,
            "version": "3.11.5",
            "path": None,
            "metadata": {},
            "issues": [],
            "recommendations": [],
        },
    },
}


@pytest.fixture
def client(monkeypatch):
    """Create a test client with an in-memory database."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    import env_doctor.server.database as db_mod
    from env_doctor.server.database import Base, get_session

    # Use in-memory SQLite for tests — must use same engine everywhere
    test_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session():
        async with test_session_factory() as session:
            yield session

    async def override_init_db():
        from env_doctor.server import models  # noqa: F401
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Patch the module-level init_db so that the lifespan uses our test engine
    monkeypatch.setattr(db_mod, "init_db", override_init_db)

    # Import app AFTER patching so lifespan picks up the override
    from env_doctor.server.app import app
    from env_doctor.server.auth import require_token

    app.dependency_overrides[get_session] = override_get_session
    # Bypass shared-token auth for these route-level tests; auth itself is
    # covered in tests/unit/server/test_auth.py.
    app.dependency_overrides[require_token] = lambda: None

    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


class TestPostReport:
    def test_creates_machine_and_snapshot(self, client):
        resp = client.post("/api/report", json=SAMPLE_REPORT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "snapshot_id" in data

    def test_upserts_machine_on_second_report(self, client):
        client.post("/api/report", json=SAMPLE_REPORT)

        # Second report with different status
        report2 = {**SAMPLE_REPORT, "status": "warning"}
        resp = client.post("/api/report", json=report2)
        assert resp.status_code == 200

        # Should still be one machine
        machines = client.get("/api/machines").json()
        assert len(machines) == 1
        assert machines[0]["latest_status"] == "warning"


class TestGetMachines:
    def test_empty_list(self, client):
        resp = client.get("/api/machines")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_machines_after_report(self, client):
        client.post("/api/report", json=SAMPLE_REPORT)
        resp = client.get("/api/machines")
        machines = resp.json()
        assert len(machines) == 1
        assert machines[0]["hostname"] == "gpu-node-01"
        assert machines[0]["gpu_name"] == "NVIDIA A100-SXM4-80GB"

    def test_filter_by_status(self, client):
        client.post("/api/report", json=SAMPLE_REPORT)
        # Add a failing machine
        fail_report = {**SAMPLE_REPORT}
        fail_report["machine"] = {**SAMPLE_REPORT["machine"], "machine_id": "test-uuid-002", "hostname": "gpu-bad"}
        fail_report["status"] = "fail"
        client.post("/api/report", json=fail_report)

        all_machines = client.get("/api/machines").json()
        assert len(all_machines) == 2

        pass_only = client.get("/api/machines?status=pass").json()
        assert len(pass_only) == 1
        assert pass_only[0]["hostname"] == "gpu-node-01"


class TestGetMachineDetail:
    def test_returns_full_report(self, client):
        client.post("/api/report", json=SAMPLE_REPORT)
        resp = client.get("/api/machines/test-uuid-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hostname"] == "gpu-node-01"
        assert "latest_report" in data
        assert data["latest_report"]["status"] == "pass"

    def test_404_for_unknown_machine(self, client):
        resp = client.get("/api/machines/nonexistent")
        assert resp.status_code == 404


class TestGetMachineHistory:
    def test_returns_snapshots(self, client):
        client.post("/api/report", json=SAMPLE_REPORT)
        report2 = {**SAMPLE_REPORT, "status": "warning", "timestamp": "2025-01-02T00:00:00"}
        client.post("/api/report", json=report2)

        resp = client.get("/api/machines/test-uuid-001/history")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) == 2
        # Most recent first
        assert history[0]["status"] == "warning"

    def test_404_for_unknown_machine(self, client):
        resp = client.get("/api/machines/nonexistent/history")
        assert resp.status_code == 404

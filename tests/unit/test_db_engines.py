"""
Tests for inference-engine data loading and the bundled inference_engines.json.
"""
import json
import os

import pytest
from packaging.version import Version

from env_doctor import db
from env_doctor.engines import render_install

BUNDLED = os.path.join(
    os.path.dirname(db.__file__), "data", "inference_engines.json"
)


@pytest.fixture(scope="module")
def engine_data():
    with open(BUNDLED, "r") as f:
        return json.load(f)


class TestMinDriverForCuda:
    @pytest.mark.parametrize("cuda,expected", [
        ("13.0", "580"),
        ("12.9", "575"),
        ("12.6", "560"),
        ("12.0", "525"),
        ("11.0", "450"),
    ])
    def test_inverts_driver_to_cuda(self, cuda, expected):
        assert db.get_min_driver_for_cuda(cuda) == expected

    def test_unknown_future_cuda(self):
        assert db.get_min_driver_for_cuda("99.0") is None

    def test_invalid(self):
        assert db.get_min_driver_for_cuda("garbage") is None

    def test_round_trip(self):
        drv = db.get_min_driver_for_cuda("12.8")
        assert Version(db.get_max_cuda_for_driver(f"{drv}.00")) >= Version("12.8")


class TestBundledEngineData:
    def test_has_vllm_and_sglang(self, engine_data):
        assert {"vllm", "sglang"} <= set(engine_data["engines"])
        assert engine_data["_metadata"]["last_verified"]

    def test_every_version_is_well_formed(self, engine_data):
        for name, spec in engine_data["engines"].items():
            templates = spec["install_templates"]
            for ver, rec in spec["versions"].items():
                Version(ver)  # parseable
                assert rec.get("torch"), f"{name} {ver}: missing torch pin"
                assert rec["default_cuda"] in rec["cuda_builds"], f"{name} {ver}: default build missing"
                for build, tmpl in rec["cuda_builds"].items():
                    Version(build)
                    assert tmpl in templates, f"{name} {ver}: unknown template {tmpl}"
                    cmds = render_install(spec, ver, build)
                    assert cmds, f"{name} {ver} cu{build}: renders no commands"
                    assert all("{" not in c for c in cmds)
                    assert any(f"=={ver}" in c for c in cmds)

    def test_vllm_default_flips_to_cuda13_at_0_20(self, engine_data):
        """Seed-data regression guard for the issue #139 trap."""
        versions = engine_data["engines"]["vllm"]["versions"]
        assert versions["0.20.0"]["default_cuda"] == "13.0"
        assert versions["0.19.1"]["default_cuda"] == "12.9"


class TestLoadEngineData:
    def test_falls_back_to_bundled_when_offline(self, monkeypatch, tmp_path):
        import requests

        def boom(*a, **k):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr(db, "_ENGINE_DATA", None)
        monkeypatch.setattr(db, "ENGINE_CACHE_FILE", str(tmp_path / "missing.json"))
        monkeypatch.setattr(db.requests, "get", boom)

        data = db.load_engine_data()
        assert "vllm" in data["engines"]
        monkeypatch.setattr(db, "_ENGINE_DATA", None)

    def test_uses_fresh_cache(self, monkeypatch, tmp_path):
        cache = tmp_path / "engines.json"
        cache.write_text(json.dumps({"engines": {"cached": {}}}))
        monkeypatch.setattr(db, "_ENGINE_DATA", None)
        monkeypatch.setattr(db, "ENGINE_CACHE_FILE", str(cache))

        assert "cached" in db.load_engine_data()["engines"]
        monkeypatch.setattr(db, "_ENGINE_DATA", None)

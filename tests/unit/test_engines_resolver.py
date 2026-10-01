"""
Unit tests for the inference-engine resolver (env_doctor.engines).

Uses a small fixture DB so tests don't move when the shipped
inference_engines.json is updated.
"""
import pytest

from env_doctor.core.detector import Status
from env_doctor.engines import (
    COMPAT_MINOR,
    COMPAT_NO,
    COMPAT_OK,
    build_compat,
    cuda_from_local_tag,
    parse_engine_spec,
    render_install,
    resolve_engine,
    sorted_versions,
)

MIN_DRIVER = {"11.0": "450", "12.0": "525", "12.6": "560", "12.9": "575", "13.0": "580"}


def min_driver(cuda):
    return MIN_DRIVER.get(cuda)


ENGINE_DB = {
    "_metadata": {"last_verified": "2026-10-01"},
    "engines": {
        "vllm": {
            "display_name": "vLLM",
            "pip_name": "vllm",
            "install_templates": {
                "pypi": ["uv pip install vllm=={version} --torch-backend={cu_tag}"],
                "vllm_index": [
                    "uv pip install vllm=={version} --extra-index-url "
                    "https://wheels.vllm.ai/{version}/{cu_tag} --torch-backend={cu_tag}"
                ],
            },
            "versions": {
                "0.30.0": {"torch": "2.13.0", "default_cuda": "13.0",
                           "cuda_builds": {"13.0": "pypi", "12.9": "vllm_index"},
                           "kernel_libs": {"flashinfer-python": "0.6.18"}},
                "0.19.1": {"torch": "2.10.0", "default_cuda": "12.9",
                           "cuda_builds": {"12.9": "pypi", "13.0": "vllm_index"},
                           "kernel_libs": {"flashinfer-python": "0.6.6"}},
                "0.9.0": {"torch": "2.7.0", "default_cuda": "12.6",
                          "cuda_builds": {"12.6": "pypi"}},
            },
        },
        "sglang": {
            "display_name": "SGLang",
            "pip_name": "sglang",
            "install_templates": {
                "pypi": ["uv pip install sglang=={version}"],
                "cu12_lane": [
                    "uv pip install sglang=={version}",
                    "uv pip install --force-reinstall torch=={torch} --index-url https://download.pytorch.org/whl/{cu_tag}",
                    "uv pip install --force-reinstall sglang-kernel=={sglang_kernel} --index-url https://docs.sglang.ai/whl/{cu_tag}/",
                    "uv pip install --force-reinstall sgl-deep-gemm=={sgl_deep_gemm} --index-url https://docs.sglang.ai/whl/{cu_tag}/ --no-deps",
                ],
            },
            "versions": {
                "0.5.21": {"torch": "2.13.0", "default_cuda": "13.0", "cuda_builds": {"13.0": "pypi"},
                           "kernel_libs": {"sglang-kernel": "0.4.7"}},
                "0.5.19": {"torch": "2.13.0", "default_cuda": "13.0",
                           "cuda_builds": {"13.0": "pypi", "12.9": "cu12_lane"},
                           "kernel_libs": {"sglang-kernel": "0.4.6.post1", "sgl-deep-gemm": "0.1.7"},
                           "notes": "Last release with a CUDA 12 lane."},
                "0.5.11": {"torch": "2.11.0", "default_cuda": "13.0",
                           "cuda_builds": {"13.0": "pypi", "12.9": "cu12_lane"},
                           "kernel_libs": {"sglang-kernel": "0.4.2"}},
            },
        },
    },
}


def resolve(engine, max_cuda, **kw):
    kw.setdefault("min_driver_for_cuda", min_driver)
    return resolve_engine(engine, max_cuda, ENGINE_DB, **kw)


# ===== Helpers =====

class TestHelpers:
    @pytest.mark.parametrize("spec,expected", [
        ("vllm", ("vllm", None)),
        ("vLLM@0.20.0", ("vllm", "0.20.0")),
        ("sglang==0.5.19", ("sglang", "0.5.19")),
    ])
    def test_parse_engine_spec(self, spec, expected):
        assert parse_engine_spec(spec) == expected

    @pytest.mark.parametrize("tag,expected", [
        ("2.13.0+cu129", "12.9"),
        ("0.30.0+cu130", "13.0"),
        ("2.0.1+cu118", "11.8"),
        ("2.13.0", None),
        (None, None),
    ])
    def test_cuda_from_local_tag(self, tag, expected):
        assert cuda_from_local_tag(tag) == expected

    @pytest.mark.parametrize("build,driver,expected", [
        ("12.6", "12.6", COMPAT_OK),
        ("12.4", "13.0", COMPAT_OK),
        ("12.9", "12.6", COMPAT_MINOR),   # minor-version compatibility within 12.x
        ("13.0", "12.6", COMPAT_NO),      # major jump: the L40S failure
        ("12.10", "12.9", COMPAT_MINOR),  # version-aware, not float ("12.10" > "12.9")
    ])
    def test_build_compat(self, build, driver, expected):
        assert build_compat(build, driver) == expected

    def test_sorted_versions_uses_packaging_order(self):
        spec = {"versions": {"0.9.0": {}, "0.10.0": {}, "0.19.1": {}, "0.5.15.post1": {}}}
        assert sorted_versions(spec) == ["0.19.1", "0.10.0", "0.9.0", "0.5.15.post1"]

    def test_render_install_skips_lines_with_missing_pins(self):
        spec = ENGINE_DB["engines"]["sglang"]
        cmds = render_install(spec, "0.5.11", "12.9")  # 0.5.11 has no sgl-deep-gemm pin
        assert len(cmds) == 3
        assert not any("sgl-deep-gemm" in c for c in cmds)
        assert "sglang-kernel==0.4.2" in cmds[2]
        assert "whl/cu129" in cmds[1]


# ===== Resolver =====

class TestResolveVllm:
    def test_l40s_cuda126_canonical_case(self):
        """Issue #139: CUDA 12.6 box, latest vLLM defaults to a CUDA 13 wheel."""
        res = resolve("vllm", "12.6")

        assert res.target_version == "0.30.0"
        assert res.default_cuda == "13.0"
        assert res.status == "warning"
        assert any("CUDA 13.0" in i for i in res.issues)

        # Best fix keeps the newest engine and swaps the CUDA build — no driver change
        first = res.options[0]
        assert first.kind == "alternate_build"
        assert first.cuda == "12.9"
        assert not first.risky
        assert res.copy_to_fix == (
            "uv pip install vllm==0.30.0 --extra-index-url "
            "https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129"
        )
        # Driver upgrade is offered only last, and flagged risky
        assert res.options[-1].kind == "driver_upgrade"
        assert res.options[-1].risky
        assert "580" in res.options[-1].summary
        assert any("minor-version" in w for w in res.warnings)

    def test_driver_supports_default_build(self):
        res = resolve("vllm", "13.0")
        assert res.status == "ok"
        assert [o.kind for o in res.options] == ["default"]
        assert res.copy_to_fix == "uv pip install vllm==0.30.0 --torch-backend=cu130"
        assert not res.issues and not res.warnings

    def test_explicit_target_version(self):
        res = resolve("vllm", "12.6", target_version="0.19.1")
        assert res.target_version == "0.19.1"
        assert res.options[0].kind == "default"
        assert res.options[0].cuda == "12.9"
        assert res.copy_to_fix == "uv pip install vllm==0.19.1 --torch-backend=cu129"

    def test_no_usable_build_without_driver_upgrade(self):
        res = resolve("vllm", "11.8")
        assert res.status == "error"
        assert res.copy_to_fix is None
        assert [o.kind for o in res.options] == ["driver_upgrade"]
        assert res.options[0].risky

    def test_falls_back_to_other_version(self):
        """Target has no usable build, but an older release fits exactly."""
        db = {"engines": {"vllm": dict(ENGINE_DB["engines"]["vllm"])}}
        db["engines"]["vllm"]["versions"] = {
            "0.30.0": {"torch": "2.13.0", "default_cuda": "13.0", "cuda_builds": {"13.0": "pypi"}},
            "0.9.0": {"torch": "2.7.0", "default_cuda": "12.6", "cuda_builds": {"12.6": "pypi"}},
        }
        res = resolve_engine("vllm", "12.6", db, min_driver_for_cuda=min_driver)
        assert res.options[0].kind == "older_version"
        assert res.options[0].version == "0.9.0"
        assert res.copy_to_fix == "uv pip install vllm==0.9.0 --torch-backend=cu126"
        assert res.status == "warning"

    def test_no_driver(self):
        res = resolve("vllm", None)
        assert res.status == "unknown"
        assert res.copy_to_fix == "uv pip install vllm==0.30.0 --torch-backend=cu130"

    def test_missing_uv_prepends_install(self):
        res = resolve("vllm", "13.0", have_uv=False)
        assert res.options[0].commands[0] == "pip install uv"

    def test_unknown_engine(self):
        res = resolve("tgi", "12.6")
        assert res.status == "unknown"
        assert "not a known inference engine" in res.issues[0]


class TestResolveInstalled:
    def test_installed_cuda13_build_on_cuda12_driver_is_error(self):
        installed = {"engine_version": "0.30.0", "torch_cuda": "13.0"}
        res = resolve("vllm", "12.6", installed=installed)
        assert res.status == "error"
        assert res.installed_cuda == "13.0"
        assert any("will fail at runtime" in i for i in res.issues)
        assert res.options[0].kind == "alternate_build"
        assert "cu129" in res.copy_to_fix

    def test_installed_alternate_build_that_works(self):
        """Someone already installed vllm+cu129 on a 12.x box: no fix needed."""
        installed = {"engine_version": "0.30.0+cu129", "engine_cuda": "12.9"}
        res = resolve("vllm", "12.9", installed=installed)
        assert res.status == "ok"
        assert res.options == []
        assert res.copy_to_fix is None

    def test_installed_torch_cuda_used_as_build(self):
        installed = {"engine_version": "0.30.0", "torch_cuda": "12.9"}
        res = resolve("vllm", "12.9", installed=installed)
        assert res.status == "ok"
        assert res.torch_pin == "2.13.0"

    def test_kernel_lib_drift_warns(self):
        installed = {
            "engine_version": "0.5.19", "torch_cuda": "13.0",
            "kernel_libs": {"sglang-kernel": "0.4.5", "sgl_deep_gemm": "0.1.7+cu130"},
        }
        res = resolve("sglang", "13.0", installed=installed)
        assert res.status == "warning"
        drift = [w for w in res.warnings if "pins" in w]
        assert len(drift) == 1 and "sglang-kernel 0.4.5" in drift[0]

    def test_installed_engine_not_in_db(self):
        installed = {"engine_version": "0.99.0", "torch_cuda": "12.9"}
        res = resolve("vllm", "12.9", installed=installed)
        assert any("not in the compatibility DB" in w for w in res.warnings)
        assert res.status == "warning"


class TestResolveSglang:
    def test_latest_has_no_cuda12_lane_falls_back_to_last_lane(self):
        res = resolve("sglang", "12.6")
        assert res.target_version == "0.5.21"
        first = res.options[0]
        assert first.kind == "older_version"
        assert first.version == "0.5.19"
        assert first.cuda == "12.9"
        assert "Last release with a CUDA 12 lane" in first.summary
        assert len(first.commands) == 4
        assert res.options[-1].risky

    def test_off_default_build_not_offered_when_absent(self):
        """Strict pins: never invent a CUDA build the release doesn't publish."""
        res = resolve("sglang", "12.6", target_version="0.5.21")
        assert all(not (o.version == "0.5.21" and o.cuda == "12.9") for o in res.options)


class TestDetectionResultAdapter:
    def test_maps_status_and_carries_copy_to_fix(self):
        det = resolve("vllm", "12.6").to_detection_result()
        assert det.component == "engine_vllm"
        assert det.status == Status.WARNING
        assert det.metadata["copy_to_fix"].startswith("uv pip install vllm==0.30.0")
        assert det.recommendations[0].startswith("1. Install the CUDA 12.9 build")
        assert "(⚠ risky)" in det.recommendations[-1]

    def test_error_maps_to_error(self):
        det = resolve("vllm", "11.8").to_detection_result()
        assert det.status == Status.ERROR

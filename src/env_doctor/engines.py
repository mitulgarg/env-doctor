"""
Inference-engine (vLLM / SGLang) ↔ driver/CUDA compatibility resolver.

Answers: "given the driver on this box, which build of <engine> will actually
run, and what's the exact install command?"

The chain being resolved is:

    engine version → pinned torch → CUDA build of the wheels → kernel libs
                                         ↓
                     must run on the INSTALLED NVIDIA driver

pip wheels bundle their own CUDA runtime, so the binding constraint is the
driver's max supported CUDA, not the system toolkit (nvcc). Within one CUDA
major version, NVIDIA's minor-version compatibility lets a wheel built for a
newer minor (e.g. 12.9) run on an older 12.x driver; across majors it does not
(a CUDA 13 wheel on a 12.6 driver fails).

This module is pure logic (no detection, no I/O) so the CLI, MCP server and
Python API can all share it. Data comes from data/inference_engines.json.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from packaging.version import Version, InvalidVersion

from .core.detector import DetectionResult, Status

SUPPORTED_ENGINES = ("vllm", "sglang")

# Build compatibility levels
COMPAT_OK = "ok"                    # build CUDA <= driver max CUDA
COMPAT_MINOR = "minor_compat"       # same major, newer minor: runs via minor-version compat
COMPAT_NO = "incompatible"          # newer CUDA major than the driver supports

# Result statuses
STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"

_STATUS_MAP = {
    STATUS_OK: Status.SUCCESS,
    STATUS_WARNING: Status.WARNING,
    STATUS_ERROR: Status.ERROR,
    STATUS_UNKNOWN: Status.WARNING,
}


@dataclass
class FixOption:
    kind: str                   # "default" | "alternate_build" | "newer_version" | "older_version" | "driver_upgrade"
    summary: str
    commands: List[str] = field(default_factory=list)
    version: Optional[str] = None
    cuda: Optional[str] = None
    risky: bool = False


@dataclass
class EngineResolution:
    engine: str
    display_name: str
    status: str
    target_version: Optional[str] = None
    installed_version: Optional[str] = None
    installed_cuda: Optional[str] = None
    default_cuda: Optional[str] = None
    driver_max_cuda: Optional[str] = None
    torch_pin: Optional[str] = None
    kernel_libs: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    options: List[FixOption] = field(default_factory=list)
    copy_to_fix: Optional[str] = None
    db_last_verified: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_detection_result(self) -> DetectionResult:
        """Adapt to a DetectionResult so status/exit-code/HTML logic reuse works."""
        recs = []
        for i, opt in enumerate(self.options, 1):
            flag = " (⚠ risky)" if opt.risky else ""
            recs.append(f"{i}. {opt.summary}{flag}: {' && '.join(opt.commands)}"
                        if opt.commands else f"{i}. {opt.summary}{flag}")
        return DetectionResult(
            component=f"engine_{self.engine}",
            status=_STATUS_MAP[self.status],
            version=self.target_version or self.installed_version,
            metadata=self.to_dict(),
            issues=self.issues + self.warnings,
            recommendations=recs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_engine_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse 'vllm', 'vllm@0.20.0' or 'vllm==0.20.0' into (name, version)."""
    m = re.match(r"^\s*([A-Za-z0-9_.-]+?)\s*(?:@|==)\s*([^\s]+)\s*$", spec)
    if m:
        return m.group(1).lower(), m.group(2)
    return spec.strip().lower(), None


def cuda_tag(cuda: str) -> str:
    """'12.9' -> 'cu129'."""
    return "cu" + cuda.replace(".", "")


def cuda_from_local_tag(version: Optional[str]) -> Optional[str]:
    """Extract CUDA from a local version tag: '2.13.0+cu129' -> '12.9'."""
    if not version:
        return None
    m = re.search(r"\+cu(\d+)(\d)$", version)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


def _v(s: str) -> Optional[Version]:
    try:
        return Version(s)
    except (InvalidVersion, TypeError):
        return None


def _normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def build_compat(build_cuda: str, driver_max_cuda: str) -> str:
    """Can a wheel built for build_cuda run on a driver supporting driver_max_cuda?"""
    b, d = _v(build_cuda), _v(driver_max_cuda)
    if b is None or d is None:
        return COMPAT_NO
    if b <= d:
        return COMPAT_OK
    if b.major == d.major:
        return COMPAT_MINOR
    return COMPAT_NO


def sorted_versions(engine_spec: dict) -> List[str]:
    """Engine versions in the DB, newest first (packaging order, not float/str)."""
    vers = [v for v in engine_spec.get("versions", {}) if _v(v) is not None]
    return sorted(vers, key=Version, reverse=True)


def render_install(engine_spec: dict, version: str, build_cuda: str) -> List[str]:
    """Render the install commands for a given engine version + CUDA build."""
    record = engine_spec["versions"][version]
    template_name = record["cuda_builds"][build_cuda]
    template = engine_spec.get("install_templates", {}).get(template_name, [])

    fields = {"version": version, "cu_tag": cuda_tag(build_cuda), "torch": record.get("torch", "")}
    for pins in (record.get("kernel_libs", {}), record.get("extra_pins", {})):
        for name, ver in pins.items():
            fields[name.replace("-", "_")] = ver

    commands = []
    for line in template:
        try:
            commands.append(line.format(**fields))
        except KeyError:
            continue  # line references a pin this version doesn't have
    return commands


def _ranked_builds(record: dict, driver_max_cuda: str) -> List[Tuple[str, str]]:
    """Usable builds for a version, best first: exact-fit before minor-compat, newest CUDA first."""
    usable = []
    for build in record.get("cuda_builds", {}):
        level = build_compat(build, driver_max_cuda)
        if level != COMPAT_NO:
            usable.append((build, level))
    usable.sort(key=lambda bl: (bl[1] == COMPAT_OK, _v(bl[0]) or Version("0")), reverse=True)
    return usable


def _minor_compat_note(build: str, driver_max_cuda: str, min_driver_for_major: Optional[str]) -> str:
    drv = f" (driver ≥ {min_driver_for_major})" if min_driver_for_major else ""
    return (f"CUDA {build} build runs on a CUDA {driver_max_cuda} driver via CUDA minor-version "
            f"compatibility{drv}. Precompiled kernels work; anything needing PTX JIT for a newer "
            f"toolkit may not.")


def _with_uv(commands: List[str], have_uv: bool) -> List[str]:
    if not have_uv and any(c.startswith("uv ") for c in commands):
        return ["pip install uv"] + commands
    return commands


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_engine(engine: str,
                   driver_max_cuda: Optional[str],
                   engine_db: dict,
                   target_version: Optional[str] = None,
                   installed: Optional[Dict[str, Any]] = None,
                   toolkit_cuda: Optional[str] = None,
                   min_driver_for_cuda=None,
                   have_uv: bool = True) -> EngineResolution:
    """
    Resolve which build of an inference engine will run on this machine.

    Args:
        engine: "vllm" or "sglang".
        driver_max_cuda: Max CUDA the installed driver supports (e.g. "12.6"); None if no driver.
        engine_db: Parsed inference_engines.json.
        target_version: Version the user intends to install (None = installed, else latest).
        installed: Detector metadata for an installed engine:
            {"engine_version", "engine_cuda", "torch_version", "torch_cuda", "kernel_libs"}.
        toolkit_cuda: System nvcc version, used only for JIT warnings.
        min_driver_for_cuda: Callable(cuda) -> driver branch (db.get_min_driver_for_cuda).
        have_uv: Whether `uv` is on PATH; if not, commands are prefixed with `pip install uv`.

    Fix options are ordered best-first: same version with a different CUDA
    build, then a different engine version, and only last a driver upgrade
    (flagged risky — it can break other deployments on a shared box).
    """
    min_driver_for_cuda = min_driver_for_cuda or (lambda _c: None)
    engines = engine_db.get("engines", {})
    meta = engine_db.get("_metadata", {})

    if engine not in engines:
        known = ", ".join(sorted(engines)) or "none"
        return EngineResolution(
            engine=engine, display_name=engine, status=STATUS_UNKNOWN,
            target_version=target_version,
            issues=[f"'{engine}' is not a known inference engine (known: {known})."],
            db_last_verified=meta.get("last_verified"),
        )

    spec = engines[engine]
    pip_name = spec.get("pip_name", engine)
    versions = sorted_versions(spec)
    installed = installed or {}
    installed_version = installed.get("engine_version")
    if installed_version and "+" in installed_version:
        # "0.30.0+cu129" → DB record "0.30.0"; the tag is the CUDA build
        installed = {**installed, "engine_cuda": installed.get("engine_cuda") or cuda_from_local_tag(installed_version)}
        installed_version = installed_version.split("+", 1)[0]

    res = EngineResolution(
        engine=engine,
        display_name=spec.get("display_name", engine),
        status=STATUS_OK,
        installed_version=installed_version,
        driver_max_cuda=driver_max_cuda,
        db_last_verified=meta.get("last_verified"),
    )

    # Which version are we evaluating?
    explicit_target = target_version is not None
    target = target_version or installed_version or (versions[0] if versions else None)
    res.target_version = target
    record = spec.get("versions", {}).get(target) if target else None

    if record:
        res.default_cuda = record.get("default_cuda")
        res.torch_pin = record.get("torch")
        res.kernel_libs = dict(record.get("kernel_libs", {}))
    elif target:
        res.warnings.append(
            f"{res.display_name} {target} is not in the compatibility DB "
            f"(last verified {meta.get('last_verified', 'unknown')}); recommendations are based on known releases."
        )

    # CUDA build of the installed engine: explicit local tag > torch build > DB default
    if installed_version:
        res.installed_cuda = (
            installed.get("engine_cuda")
            or installed.get("torch_cuda")
            or (spec.get("versions", {}).get(installed_version, {}).get("default_cuda"))
        )

    # --- No driver: can't verify; just hand back the default install ---
    if not driver_max_cuda or _v(driver_max_cuda) is None:
        res.status = STATUS_UNKNOWN
        res.issues.append("No NVIDIA driver detected — cannot verify which CUDA build will run.")
        if record and res.default_cuda in record.get("cuda_builds", {}):
            cmds = _with_uv(render_install(spec, target, res.default_cuda), have_uv)
            res.options.append(FixOption("default", f"Default install ({res.display_name} {target}, CUDA {res.default_cuda})",
                                         cmds, target, res.default_cuda))
            res.copy_to_fix = " && ".join(cmds)
        return res

    # --- Installed engine: is what's on disk runnable? ---
    installed_broken = False
    evaluating_installed = installed_version is not None and target == installed_version
    if evaluating_installed and res.installed_cuda:
        level = build_compat(res.installed_cuda, driver_max_cuda)
        if level == COMPAT_NO:
            installed_broken = True
            res.issues.append(
                f"Installed {res.display_name} {installed_version} is built for CUDA {res.installed_cuda}, "
                f"but the driver supports up to CUDA {driver_max_cuda} — it will fail at runtime."
            )
        elif level == COMPAT_MINOR:
            res.warnings.append(_minor_compat_note(
                res.installed_cuda, driver_max_cuda, min_driver_for_cuda(f"{_v(res.installed_cuda).major}.0")))

    # Kernel-lib drift (the brittle part of the chain, esp. SGLang)
    if evaluating_installed and record:
        have = {_normalize_dist(k): v for k, v in (installed.get("kernel_libs") or {}).items()}
        for lib, pinned in record.get("kernel_libs", {}).items():
            got = have.get(_normalize_dist(lib))
            if got and got.split("+")[0] != pinned:
                res.warnings.append(
                    f"{lib} {got} is installed but {res.display_name} {installed_version} pins {pinned} — "
                    f"mismatched kernel libs can fail at import or produce wrong results."
                )

    # An installed engine that runs on this driver needs no fix, even if its
    # default PyPI wheel wouldn't (e.g. someone already installed the +cu129 build).
    if evaluating_installed and res.installed_cuda and not installed_broken:
        _toolkit_warning(res, toolkit_cuda, res.installed_cuda)
        res.status = STATUS_WARNING if (res.issues or res.warnings) else STATUS_OK
        return res

    # --- Is the default (naive `pip install`) build OK? ---
    default_level = None
    if record and res.default_cuda:
        default_level = build_compat(res.default_cuda, driver_max_cuda)

    if record and default_level == COMPAT_OK and not installed_broken:
        if not evaluating_installed or explicit_target:
            cmds = _with_uv(render_install(spec, target, res.default_cuda), have_uv)
            res.options.append(FixOption("default", f"Default install works (CUDA {res.default_cuda})",
                                         cmds, target, res.default_cuda))
            res.copy_to_fix = " && ".join(cmds)
        _toolkit_warning(res, toolkit_cuda, res.default_cuda)
        res.status = STATUS_WARNING if res.warnings else STATUS_OK
        return res

    if record and default_level == COMPAT_NO:
        res.issues.append(
            f"Default `pip install {pip_name}=={target}` pulls a CUDA {res.default_cuda} build (torch {res.torch_pin}), "
            f"but the driver supports up to CUDA {driver_max_cuda}."
        )

    # --- Option 1: same version, different CUDA build ---
    chosen_build = None
    if record:
        ranked = _ranked_builds(record, driver_max_cuda)
        if ranked:
            build, level = ranked[0]
            chosen_build = build
            cmds = _with_uv(render_install(spec, target, build), have_uv)
            if build == res.default_cuda:
                summary = f"Default install of {res.display_name} {target} (CUDA {build})"
                kind = "default"
            else:
                summary = f"Install the CUDA {build} build of {res.display_name} {target}"
                kind = "alternate_build"
            if level == COMPAT_MINOR:
                summary += " — via CUDA minor-version compatibility"
                res.warnings.append(_minor_compat_note(build, driver_max_cuda,
                                                       min_driver_for_cuda(f"{_v(build).major}.0")))
            res.options.append(FixOption(kind, summary, cmds, target, build))

    # --- Option 2: nearest other version with a usable build ---
    if chosen_build is None:
        alt = _best_other_version(spec, versions, target, driver_max_cuda)
        if alt:
            ver, build, level = alt
            cmds = _with_uv(render_install(spec, ver, build), have_uv)
            tv = _v(target) if target else None
            kind = "newer_version" if tv is not None and Version(ver) > tv else "older_version"
            summary = f"Use {res.display_name} {ver} — newest release with a build for this driver (CUDA {build})"
            if level == COMPAT_MINOR:
                summary += " via CUDA minor-version compatibility"
                res.warnings.append(_minor_compat_note(build, driver_max_cuda,
                                                       min_driver_for_cuda(f"{_v(build).major}.0")))
            notes = spec["versions"][ver].get("notes")
            if notes:
                summary += f". {notes}"
            res.options.append(FixOption(kind, summary, cmds, ver, build))
            chosen_build = build

    # --- Option 3 (last resort): driver upgrade ---
    need_cuda = res.default_cuda or (versions and spec["versions"][versions[0]].get("default_cuda"))
    if need_cuda and build_compat(need_cuda, driver_max_cuda) != COMPAT_OK:
        drv = min_driver_for_cuda(need_cuda)
        drv_txt = f"≥ {drv}" if drv else f"a version supporting CUDA {need_cuda}"
        res.options.append(FixOption(
            "driver_upgrade",
            f"Upgrade the NVIDIA driver to {drv_txt} to use the default CUDA {need_cuda} build "
            f"— on a shared box this can break other deployments",
            [], target, need_cuda, risky=True,
        ))

    first_safe = next((o for o in res.options if not o.risky and o.commands), None)
    if first_safe:
        res.copy_to_fix = " && ".join(first_safe.commands)
        _toolkit_warning(res, toolkit_cuda, first_safe.cuda)

    if installed_broken or first_safe is None:
        res.status = STATUS_ERROR
    elif res.issues or res.warnings:
        res.status = STATUS_WARNING
    return res


def _best_other_version(spec: dict, versions: List[str], target: Optional[str],
                        driver_max_cuda: str) -> Optional[Tuple[str, str, str]]:
    """Newest version (≠ target) that has any usable build for this driver."""
    for ver in versions:
        if ver == target:
            continue
        ranked = _ranked_builds(spec["versions"][ver], driver_max_cuda)
        if ranked:
            build, level = ranked[0]
            return ver, build, level
    return None


def _toolkit_warning(res: EngineResolution, toolkit_cuda: Optional[str], build_cuda: Optional[str]) -> None:
    """nvcc only matters for JIT-compiled kernels; flag a major-version mismatch softly."""
    t, b = _v(toolkit_cuda) if toolkit_cuda else None, _v(build_cuda) if build_cuda else None
    if t is not None and b is not None and t.major != b.major:
        res.warnings.append(
            f"System nvcc is CUDA {toolkit_cuda} but the wheels are CUDA {build_cuda}. Prebuilt wheels are fine; "
            f"JIT-compiled kernels (e.g. FlashInfer JIT) may need a matching CUDA_HOME."
        )

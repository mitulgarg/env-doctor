"""
Detector for Python version compatibility with AI libraries.

Checks if the current Python version is compatible with installed
AI libraries and detects dependency cascades that force version constraints.
"""
import sys

from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry
from env_doctor.db import PYTHON_COMPAT_DATA


def _version_tuple(version_str: str):
    """Convert '3.11' to (3, 11) for comparison."""
    parts = version_str.strip().split(".")
    return tuple(int(p) for p in parts)


def _is_library_installed(import_name: str) -> bool:
    """Check if a library is installed using importlib.metadata."""
    try:
        import importlib.metadata
        # Try finding by distribution name first
        try:
            importlib.metadata.distribution(import_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            pass
        # Some packages have different distribution vs import names
        # Try import as fallback
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False
    except Exception:
        return False


@DetectorRegistry.register("python_compat")
class PythonCompatDetector(Detector):
    """Detects Python version compatibility issues with AI libraries."""

    def detect(self) -> DetectionResult:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        py_tuple = (sys.version_info.major, sys.version_info.minor)

        constraints = PYTHON_COMPAT_DATA.get("python_version_constraints", [])
        cascades = PYTHON_COMPAT_DATA.get("dependency_cascades", [])

        conflicts = []
        installed_import_names = set()

        for constraint in constraints:
            library = constraint.get("library", "")
            import_name = constraint.get("import_name", library)

            if not _is_library_installed(import_name):
                continue

            installed_import_names.add(import_name)

            min_ver = constraint.get("min_version")
            max_ver = constraint.get("max_version")
            status = constraint.get("status", "active")

            if status != "active":
                continue

            conflict_info = {
                "library": library,
                "min_version": min_ver,
                "max_version": max_ver,
                "notes": constraint.get("notes", ""),
            }

            if min_ver and py_tuple < _version_tuple(min_ver):
                conflict_info["type"] = "below_minimum"
                conflict_info["message"] = (
                    f"{library} requires Python >={min_ver}, "
                    f"but you have Python {py_version}"
                )
                conflicts.append(conflict_info)
            elif max_ver and py_tuple > _version_tuple(max_ver):
                conflict_info["type"] = "above_maximum"
                conflict_info["message"] = (
                    f"{library} supports Python <={max_ver}, "
                    f"but you have Python {py_version}"
                )
                conflicts.append(conflict_info)

        # Find cascade impacts for conflicting libraries
        cascade_impacts = []
        conflicting_libs = {c["library"] for c in conflicts}
        for cascade in cascades:
            root = cascade.get("root_library", "")
            if root in conflicting_libs:
                cascade_impacts.append({
                    "root_library": root,
                    "affected_dependencies": cascade.get("affected_dependencies", []),
                    "severity": cascade.get("severity", "medium"),
                    "description": cascade.get("description", ""),
                })

        # Build result
        issues = [c["message"] for c in conflicts]
        recommendations = []

        if conflicts:
            below = [c for c in conflicts if c["type"] == "below_minimum"]
            above = [c for c in conflicts if c["type"] == "above_maximum"]

            if below:
                min_needed = max(
                    _version_tuple(c["min_version"]) for c in below
                )
                min_str = ".".join(str(x) for x in min_needed)
                recommendations.append(
                    f"Upgrade Python to at least {min_str} for compatibility"
                )
            if above:
                max_allowed = min(
                    _version_tuple(c["max_version"]) for c in above
                )
                max_str = ".".join(str(x) for x in max_allowed)
                recommendations.append(
                    f"Consider using Python {max_str} or lower for full compatibility"
                )

        if cascade_impacts:
            for cascade in cascade_impacts:
                affected = ", ".join(cascade["affected_dependencies"])
                recommendations.append(
                    f"Cascade: {cascade['root_library']} constraint also affects: {affected}"
                )

        # Determine status
        if conflicts:
            result_status = Status.ERROR
        elif cascade_impacts:
            result_status = Status.WARNING
        else:
            result_status = Status.SUCCESS

        return DetectionResult(
            component="python_compat",
            status=result_status,
            version=py_version,
            metadata={
                "python_full_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "conflicts": conflicts,
                "cascades": cascade_impacts,
                "constraints_checked": len(installed_import_names),
            },
            issues=issues,
            recommendations=recommendations,
        )

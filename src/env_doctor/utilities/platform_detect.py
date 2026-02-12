"""
Platform detection for CUDA Toolkit installation recommendations.

Detects OS, distribution, version, and architecture to match
against cuda_toolkit_install.json platform keys.
"""
import platform
import os
from typing import Dict


def detect_platform() -> Dict[str, str]:
    """
    Detect the current platform for matching installation instructions.

    Returns:
        Dict with keys: os, distro, distro_version, arch, is_wsl2, platform_key, platform_keys
    """
    system = platform.system()  # "Linux", "Windows", "Darwin"
    arch = platform.machine()   # "x86_64", "aarch64", "AMD64"

    # Normalize architecture
    if arch in ("AMD64", "x86_64"):
        arch = "x86_64"
    elif arch in ("aarch64", "arm64"):
        arch = "aarch64"

    info = {
        "os": system.lower(),
        "arch": arch,
        "distro": None,
        "distro_version": None,
        "is_wsl2": False,
        "platform_key": None,
        "platform_keys": []  # Multiple keys to try, in priority order
    }

    if system == "Linux":
        info["is_wsl2"] = _detect_wsl2()
        distro_info = _detect_linux_distro()
        info["distro"] = distro_info.get("id", "unknown")
        info["distro_version"] = distro_info.get("version", "unknown")

        if info["is_wsl2"]:
            # WSL2 gets a special platform key (highest priority)
            info["platform_keys"].append(f"linux_wsl2_{info['distro']}")

        # Standard Linux key with full version
        info["platform_keys"].append(
            f"linux_{info['distro']}_{info['distro_version']}_{arch}"
        )

        # Try major version only (e.g., linux_ubuntu_22_x86_64)
        if "." in str(info["distro_version"]):
            major_version = info["distro_version"].split(".")[0]
            info["platform_keys"].append(
                f"linux_{info['distro']}_{major_version}_{arch}"
            )

    elif system == "Windows":
        win_version = platform.version()  # e.g., "10.0.22621"
        major = win_version.split(".")[0]
        info["distro"] = "windows"
        info["distro_version"] = major
        info["platform_keys"].append(f"windows_10_11_{arch}")

    elif system == "Darwin":
        # macOS - CUDA deprecated but include for completeness
        mac_version = platform.mac_ver()[0]  # e.g., "14.0"
        info["distro"] = "macos"
        info["distro_version"] = mac_version
        info["platform_keys"].append(f"darwin_{arch}")

    # Always add conda as final fallback
    info["platform_keys"].append("conda_any")

    info["platform_key"] = info["platform_keys"][0] if info["platform_keys"] else None

    return info


def _detect_wsl2() -> bool:
    """Check if running inside WSL2."""
    # Check /proc/version for Microsoft signature
    try:
        with open("/proc/version", "r") as f:
            content = f.read().lower()
            return "microsoft" in content
    except (FileNotFoundError, PermissionError, IOError):
        return False


def _detect_linux_distro() -> Dict[str, str]:
    """
    Detect Linux distribution from /etc/os-release.

    Returns:
        Dict with 'id' (e.g., 'ubuntu') and 'version' (e.g., '22.04')
    """
    info = {"id": "unknown", "version": "unknown"}

    # Try /etc/os-release first (modern standard)
    os_release_paths = ["/etc/os-release", "/usr/lib/os-release"]

    for path in os_release_paths:
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ID="):
                        info["id"] = line.split("=", 1)[1].strip('"').lower()
                    elif line.startswith("VERSION_ID="):
                        info["version"] = line.split("=", 1)[1].strip('"')
            if info["id"] != "unknown":
                break
        except (FileNotFoundError, PermissionError, IOError):
            continue

    # Fallback: try lsb_release if os-release didn't work
    if info["id"] == "unknown":
        try:
            import subprocess
            result = subprocess.run(
                ["lsb_release", "-is"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["id"] = result.stdout.strip().lower()

            result = subprocess.run(
                ["lsb_release", "-rs"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    return info

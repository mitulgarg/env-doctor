#!/usr/bin/env python3
"""Seed the env-doctor dashboard with a realistic synthetic fleet.

Useful for demos, screenshots, and exercising the dashboard UI (groups,
multi-select, topology bubbles, activity log) without standing up 15+ real
GPU boxes. POSTs synthetic check reports to /api/report so each machine
appears with a different GPU, driver, CUDA version, and status.

Usage:
    python scripts/seed_demo_fleet.py
    python scripts/seed_demo_fleet.py --url http://dashboard:8765
    python scripts/seed_demo_fleet.py --token <api-token>
    python scripts/seed_demo_fleet.py --reset    # delete the seeded machines
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_URL = "http://localhost:8765"
TOKEN_FILE = Path.home() / ".env-doctor" / "api-token"
SEED_PREFIX = "demo-fleet:"  # Stable machine_id prefix so --reset can find them.


# (hostname, platform, gpu_name, sm_arch, driver, max_cuda, sys_cuda, torch_version, status)
# 16 machines a mid-size MLOps team would realistically deploy.
FLEET = [
    # --- Training cluster (group these as "training-prod") ---
    ("train-h100-01",      "Linux",   "NVIDIA H100 80GB HBM3",     "9.0", "550.54.15",  "12.4", "12.4.1", "2.5.1+cu124", "pass"),
    ("train-h100-02",      "Linux",   "NVIDIA H100 80GB HBM3",     "9.0", "550.54.15",  "12.4", "12.4.1", "2.5.1+cu124", "pass"),
    ("train-a100-01",      "Linux",   "NVIDIA A100-SXM4-80GB",     "8.0", "535.146.02", "12.2", "12.2.0", "2.4.0+cu121", "pass"),
    ("train-a100-02",      "Linux",   "NVIDIA A100-SXM4-80GB",     "8.0", "535.146.02", "12.2", "12.2.0", "2.4.0+cu121", "pass"),
    ("train-a100-03",      "Linux",   "NVIDIA A100-SXM4-40GB",     "8.0", "535.146.02", "12.2", "12.2.0", "2.4.0+cu121", "warning"),

    # --- Inference fleet (group as "inference-prod") ---
    ("infer-l40s-01",      "Linux",   "NVIDIA L40S",               "8.9", "545.23.08",  "12.3", "12.3.2", "2.5.0+cu121", "pass"),
    ("infer-l40s-02",      "Linux",   "NVIDIA L40S",               "8.9", "545.23.08",  "12.3", "12.3.2", "2.5.0+cu121", "pass"),
    ("infer-a10g-01",      "Linux",   "NVIDIA A10G",               "8.6", "545.23.08",  "12.3", "12.3.2", "2.5.0+cu121", "pass"),
    ("infer-a10g-02",      "Linux",   "NVIDIA A10G",               "8.6", "545.23.08",  "12.3", "12.3.2", "2.5.0+cu121", "pass"),
    ("infer-a10g-03",      "Linux",   "NVIDIA A10G",               "8.6", "545.23.08",  "12.3", None,     "2.5.0+cu121", "fail"),

    # --- Engineer dev workstations (group as "dev") ---
    ("dev-rtx4090-mitul",  "Windows", "NVIDIA GeForce RTX 4090",   "8.9", "553.62",     "12.6", "12.4.1", "2.5.1+cu124", "pass"),
    ("dev-rtx3090-tharun", "Linux",   "NVIDIA GeForce RTX 3090",   "8.6", "535.146.02", "12.2", "12.2.0", "2.4.0+cu121", "pass"),
    ("dev-rtx3080-alex",   "Windows", "NVIDIA GeForce RTX 3080",   "8.6", "535.146.02", "12.2", "11.8.0", "2.4.0+cu121", "warning"),

    # --- Edge / research (group as "research") ---
    ("edge-l4-01",         "Linux",   "NVIDIA L4",                 "8.9", "545.23.08",  "12.3", "12.3.2", "2.5.0+cu121", "pass"),
    ("research-rtx5090-1", "Linux",   "NVIDIA GeForce RTX 5090",  "12.0", "560.94",     "12.6", "12.6.0", "2.5.1+cu124", "fail"),
    ("research-rtx5090-2", "Linux",   "NVIDIA GeForce RTX 5090",  "12.0", "560.94",     "12.6", "12.6.0", "2.5.1+cu124", "fail"),
]


def stable_machine_id(hostname: str) -> str:
    """Deterministic UUID per hostname so re-running the seeder updates instead of duplicates."""
    # uuid5 with the SEED_PREFIX namespace gives us a stable id that's unique to this hostname.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, SEED_PREFIX + hostname))


def build_report(host, plat, gpu, sm, driver, max_cuda, sys_cuda, torch_ver, status):
    """Synthesize a check report payload that the dashboard will accept."""
    now = datetime.now(timezone.utc).isoformat()

    issues = []
    if status == "warning":
        issues.append({"severity": "warning", "message": f"PyTorch built for CUDA {torch_ver.split('+cu')[-1][:2]}.x but system nvcc is {sys_cuda or 'missing'}"})
    if status == "fail":
        if "5090" in gpu:
            issues.append({"severity": "error", "message": f"PyTorch {torch_ver} does not include sm_{sm.replace('.', '')} (Blackwell). torch.cuda.is_available() may return False."})
        else:
            issues.append({"severity": "error", "message": "CUDA toolkit not found in PATH; pip-installed torch wheel cannot find nvcc."})

    checks = {
        "driver": {
            "status": "pass",
            "version": driver,
            "metadata": {
                "primary_gpu_name": gpu,
                "max_cuda_version": max_cuda,
                "gpu_count": 1,
            },
        },
        "cuda": {
            "status": "pass" if sys_cuda else "fail",
            "version": sys_cuda,
            "metadata": {"installation_count": 1 if sys_cuda else 0},
        },
        "cudnn": {"status": "pass", "version": "9.1.0"},
        "wsl2": {"status": "pass" if plat == "Linux" else "skipped"},
        "python_compat": {"status": "pass"},
        "libraries": {
            "torch": {
                "status": "pass" if status != "fail" else "fail",
                "version": torch_ver,
                "metadata": {"cuda_version": torch_ver.split("+cu")[-1] if "+cu" in torch_ver else None},
            },
        },
        "compute_compatibility": {
            "gpu_name": gpu,
            "sm": sm,
            "arch_name": _arch_for_sm(sm),
            "status": "compatible" if status != "fail" or "5090" not in gpu else "incompatible",
        },
    }

    return {
        "machine": {
            "machine_id": stable_machine_id(host),
            "hostname": host,
            "platform": plat,
            "platform_release": "5.15.0-demo" if plat == "Linux" else "10.0.22621",
            "python_version": "3.11.7",
            "reported_at": now,
        },
        "status": status,
        "timestamp": now,
        "summary": {
            "driver": driver,
            "cuda": sys_cuda or "not installed",
            "issues_count": len(issues),
            "issues": issues,
        },
        "checks": checks,
    }


def _arch_for_sm(sm: str) -> str:
    return {
        "8.0": "Ampere",
        "8.6": "Ampere",
        "8.9": "Ada Lovelace",
        "9.0": "Hopper",
        "12.0": "Blackwell",
    }.get(sm, "Unknown")


def post(url, token, machine):
    """POST one report. Returns (ok, message)."""
    body = json.dumps(machine).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/report",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except urllib.error.URLError as e:
        return False, f"URLError: {e.reason}"


def delete(url, token, machine_id):
    """DELETE one machine. Best-effort; not all server versions expose this."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/machines/{machine_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, str(e.reason)


def resolve_token(cli_token):
    if cli_token:
        return cli_token
    env = os.environ.get("ENV_DOCTOR_API_TOKEN")
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    print(f"ERROR: no token provided and {TOKEN_FILE} not found.", file=sys.stderr)
    print("Run `env-doctor dashboard` once to generate one, or pass --token.", file=sys.stderr)
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Dashboard URL (default: {DEFAULT_URL})")
    parser.add_argument("--token", help="API token (default: read ~/.env-doctor/api-token)")
    parser.add_argument("--reset", action="store_true", help="Delete the seeded fleet instead of creating it")
    args = parser.parse_args()

    token = resolve_token(args.token)

    if args.reset:
        print(f"Removing seeded fleet from {args.url}…")
        for entry in FLEET:
            host = entry[0]
            mid = stable_machine_id(host)
            ok, msg = delete(args.url, token, mid)
            mark = "OK" if ok else "skip"
            print(f"  [{mark}] {host}: {msg}")
        return

    print(f"Seeding {len(FLEET)} machines into {args.url}…")
    fail = 0
    for entry in FLEET:
        host = entry[0]
        report = build_report(*entry)
        ok, msg = post(args.url, token, report)
        mark = "OK " if ok else "ERR"
        print(f"  [{mark}] {host:24s} {entry[2]:32s} status={entry[8]:7s} {msg}")
        if not ok:
            fail += 1

    if fail:
        print(f"\n{fail}/{len(FLEET)} reports failed. Check token + URL.", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone. Open {args.url} and head to Topology to start grouping.")


if __name__ == "__main__":
    main()

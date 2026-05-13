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


def _check(component, status, version=None, path=None, metadata=None,
           issues=None, recommendations=None):
    """Build a CheckResult dict matching what real env-doctor produces.

    The dashboard frontend (DiagnosticCard.tsx) reads `issues.length` and
    `recommendations.length` directly — these MUST always be lists, never
    omitted, or React throws and the MachineDetail page renders blank.
    `status` uses the success/warning/error/not_found vocabulary, not
    pass/fail (those are the snapshot-level values).
    """
    return {
        "component": component,
        "status": status,
        "detected": status in ("success", "warning"),
        "version": version,
        "path": path,
        "metadata": metadata or {},
        "issues": issues or [],
        "recommendations": recommendations or [],
    }


def build_report(host, plat, gpu, sm, driver, max_cuda, sys_cuda, torch_ver, status):
    """Synthesize a check report payload that the dashboard will accept."""
    now = datetime.now(timezone.utc).isoformat()
    torch_cuda = torch_ver.split("+cu")[-1] if "+cu" in torch_ver else None

    # Per-check status + issue/recommendation lists, derived from machine status.
    cuda_issues, cuda_recs = [], []
    cuda_status = "success"
    if not sys_cuda:
        cuda_status = "not_found"
        cuda_issues.append("CUDA toolkit not found in PATH; nvcc unavailable.")
        cuda_recs.append("env-doctor cuda-install")

    torch_issues, torch_recs = [], []
    torch_status = "success"
    if status == "fail" and "5090" in gpu:
        torch_status = "error"
        torch_issues.append(
            f"PyTorch {torch_ver} does not include sm_{sm.replace('.', '')} (Blackwell). "
            "torch.cuda.is_available() may return False."
        )
        torch_recs.append(
            "pip install --pre torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu126"
        )
    elif status == "warning":
        torch_status = "warning"
        torch_issues.append(
            f"PyTorch built for CUDA {torch_cuda} but system nvcc is "
            f"{sys_cuda or 'missing'}."
        )

    summary_issues = []
    for c in (cuda_issues, torch_issues):
        summary_issues.extend(c)

    checks = {
        "driver": _check(
            "nvidia_driver", "success", version=driver,
            metadata={
                "primary_gpu_name": gpu,
                "max_cuda_version": max_cuda,
                "gpu_count": 1,
            },
        ),
        "cuda": _check(
            "cuda_toolkit", cuda_status, version=sys_cuda,
            metadata={"installation_count": 1 if sys_cuda else 0},
            issues=cuda_issues, recommendations=cuda_recs,
        ),
        "cudnn": _check("cudnn", "success", version="9.1.0"),
        "wsl2": _check("wsl2", "success" if plat == "Linux" else "not_found"),
        "python_compat": _check("python_compat", "success"),
        "libraries": {
            "torch": _check(
                "torch", torch_status, version=torch_ver,
                metadata={"cuda_version": torch_cuda},
                issues=torch_issues, recommendations=torch_recs,
            ),
        },
        "compute_compatibility": {
            "gpu_name": gpu,
            "compute_capability": sm,
            "sm": sm,
            "arch_name": _arch_for_sm(sm),
            "arch_list": [],
            "status": "compatible" if not (status == "fail" and "5090" in gpu) else "incompatible",
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
            "cudnn": "9.1.0",
            "issues_count": len(summary_issues),
            "issues": summary_issues,
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


# Sending Connection: close keeps urllib and uvicorn's Windows ProactorEventLoop
# in agreement on socket teardown — without it, every request prints a benign
# but noisy WinError 10054 / ConnectionResetError trace from the server side.
_BASE_HEADERS = {"Connection": "close"}


def post(url, token, machine):
    """POST one report. Returns (ok, message)."""
    body = json.dumps(machine).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/report",
        data=body,
        method="POST",
        headers={
            **_BASE_HEADERS,
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
        headers={**_BASE_HEADERS, "Authorization": f"Bearer {token}"},
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

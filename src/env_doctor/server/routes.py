"""API route handlers for the dashboard."""
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .models import Machine, Snapshot

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class MachineInfo(BaseModel):
    machine_id: str
    hostname: str
    platform: Optional[str] = None
    platform_release: Optional[str] = None
    python_version: Optional[str] = None
    reported_at: Optional[str] = None


class ReportPayload(BaseModel):
    machine: MachineInfo
    status: str
    timestamp: str
    summary: dict
    checks: dict
    heartbeat: Optional[bool] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Helper: extract denormalized fields from report payload
# ---------------------------------------------------------------------------

def _extract_fields(payload: ReportPayload) -> dict:
    """Extract denormalized fields from the report for fast queries."""
    checks = payload.checks or {}
    driver = checks.get("driver") or {}
    cuda = checks.get("cuda") or {}
    libs = checks.get("libraries") or {}
    torch_info = libs.get("torch") or {}

    gpu_name = None
    driver_meta = driver.get("metadata") or {}
    if driver_meta:
        gpu_name = driver_meta.get("primary_gpu_name")

    return {
        "summary_driver": payload.summary.get("driver"),
        "summary_cuda": payload.summary.get("cuda"),
        "summary_issues_count": payload.summary.get("issues_count", 0),
        "gpu_name": gpu_name,
        "driver_version": driver.get("version"),
        "cuda_version": cuda.get("version"),
        "torch_version": torch_info.get("version"),
    }


# ---------------------------------------------------------------------------
# POST /api/report
# ---------------------------------------------------------------------------

@router.post("/report")
async def receive_report(
    payload: ReportPayload,
    session: AsyncSession = Depends(get_session),
):
    """Receive a check report from a machine."""
    machine_id = payload.machine.machine_id
    now = datetime.now(timezone.utc)

    # Parse timestamp from payload
    try:
        ts = datetime.fromisoformat(payload.timestamp)
    except (ValueError, TypeError):
        ts = now

    # Upsert machine
    machine = await session.get(Machine, machine_id)
    if machine is None:
        machine = Machine(
            id=machine_id,
            hostname=payload.machine.hostname,
            platform=payload.machine.platform,
            python_version=payload.machine.python_version,
            first_seen=now,
            last_seen=now,
            latest_status=payload.status,
        )
        session.add(machine)
    else:
        machine.hostname = payload.machine.hostname
        machine.platform = payload.machine.platform
        machine.python_version = payload.machine.python_version
        machine.last_seen = now
        machine.latest_status = payload.status

    # Create snapshot
    fields = _extract_fields(payload)
    snapshot = Snapshot(
        machine_id=machine_id,
        status=payload.status,
        timestamp=ts,
        report_json=json.dumps(payload.model_dump(), default=str),
        is_heartbeat=bool(payload.heartbeat),
        **fields,
    )
    session.add(snapshot)
    await session.flush()

    machine.latest_snapshot_id = snapshot.id
    await session.commit()

    return {"ok": True, "snapshot_id": snapshot.id}


# ---------------------------------------------------------------------------
# GET /api/machines
# ---------------------------------------------------------------------------

@router.get("/machines")
async def list_machines(
    status: Optional[str] = Query(None, description="Filter by status: pass, warning, fail"),
    sort: str = Query("last_seen", description="Sort by field"),
    session: AsyncSession = Depends(get_session),
):
    """Return list of machines with latest status and key fields."""
    query = select(Machine)
    if status:
        query = query.where(Machine.latest_status == status)

    # Sort
    sort_col = getattr(Machine, sort, Machine.last_seen)
    query = query.order_by(sort_col.desc())

    result = await session.execute(query)
    machines = result.scalars().all()

    # For each machine, get latest snapshot denormalized fields
    output = []
    for m in machines:
        item = m.to_dict()
        if m.latest_snapshot_id:
            snap = await session.get(Snapshot, m.latest_snapshot_id)
            if snap:
                item["gpu_name"] = snap.gpu_name
                item["driver_version"] = snap.driver_version
                item["cuda_version"] = snap.cuda_version
                item["torch_version"] = snap.torch_version
        output.append(item)

    return output


# ---------------------------------------------------------------------------
# GET /api/machines/{id}
# ---------------------------------------------------------------------------

@router.get("/machines/{machine_id}")
async def get_machine(
    machine_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Return machine info + latest snapshot's full report."""
    machine = await session.get(Machine, machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    result = machine.to_dict()

    if machine.latest_snapshot_id:
        snap = await session.get(Snapshot, machine.latest_snapshot_id)
        if snap:
            result["latest_report"] = json.loads(snap.report_json)
            result["gpu_name"] = snap.gpu_name
            result["driver_version"] = snap.driver_version
            result["cuda_version"] = snap.cuda_version
            result["torch_version"] = snap.torch_version

    return result


# ---------------------------------------------------------------------------
# GET /api/machines/{id}/history
# ---------------------------------------------------------------------------

@router.get("/machines/{machine_id}/history")
async def get_machine_history(
    machine_id: str,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Return list of snapshots for a machine (timeline)."""
    machine = await session.get(Machine, machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    query = (
        select(Snapshot)
        .where(Snapshot.machine_id == machine_id)
        .order_by(Snapshot.timestamp.desc())
        .limit(limit)
    )
    result = await session.execute(query)
    snapshots = result.scalars().all()

    return [s.to_dict(include_report_json=False) for s in snapshots]

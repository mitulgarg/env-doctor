"""API route handlers for the dashboard."""
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .models import Command, Machine, Snapshot

router = APIRouter()


# A machine is "stale" when its last_seen is older than this many seconds.
# Default 1 hour = 2× the host CLI's default heartbeat interval (30m).
_STALE_AFTER_SECONDS = int(os.environ.get("ENV_DOCTOR_STALE_SECONDS", "3600"))


def _seconds_since(when: Optional[datetime]) -> Optional[float]:
    """Seconds elapsed since ``when``. Treats naive datetimes as UTC."""
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds()


# ---------------------------------------------------------------------------
# Group name validation
# ---------------------------------------------------------------------------

# Groups must be human-readable identifiers safe for URLs, file paths, and
# topology labels. Allow alphanumerics, hyphen, underscore, dot, and space.
_GROUP_NAME_PATTERN = re.compile(r"^[\w\-. ]+$")
_UNGROUPED_LABEL = "ungrouped"


def _clean_group_name(raw: Optional[str]) -> Optional[str]:
    """Normalise a group name. Returns None for empty/whitespace input.

    Raises HTTPException(400) on invalid characters so callers do not have to
    validate separately.
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if len(cleaned) > 64:
        raise HTTPException(status_code=400, detail="group_name must be 64 characters or fewer")
    if not _GROUP_NAME_PATTERN.match(cleaned):
        raise HTTPException(
            status_code=400,
            detail="group_name may only contain letters, numbers, spaces, '-', '_', '.'",
        )
    if cleaned.lower() == _UNGROUPED_LABEL:
        # Reserved synthetic label used by GET /api/groups for NULL machines.
        raise HTTPException(status_code=400, detail=f"'{_UNGROUPED_LABEL}' is a reserved name")
    return cleaned


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
    group_name: Optional[str] = None  # Optional self-tag from CLI; dashboard PATCH overrides.


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
            group_name=_clean_group_name(payload.machine.group_name),
        )
        session.add(machine)
    else:
        machine.hostname = payload.machine.hostname
        machine.platform = payload.machine.platform
        machine.python_version = payload.machine.python_version
        machine.last_seen = now
        machine.latest_status = payload.status
        # Only honour CLI self-tag when no group was set via the dashboard PATCH.
        # Dashboard-assigned groups are the source of truth and must not be
        # silently overwritten by every check-in.
        if machine.group_name is None and payload.machine.group_name:
            machine.group_name = _clean_group_name(payload.machine.group_name)

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
    await session.flush()

    # Pick up any pending commands for this machine and mark as running
    cmd_query = (
        select(Command)
        .where(Command.machine_id == machine_id, Command.status == "pending")
        .order_by(Command.created_at)
    )
    cmd_result = await session.execute(cmd_query)
    pending_cmds = cmd_result.scalars().all()
    for cmd in pending_cmds:
        cmd.status = "running"

    await session.commit()

    return {
        "ok": True,
        "snapshot_id": snapshot.id,
        "pending_commands": [c.to_dict() for c in pending_cmds],
    }


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
        elapsed = _seconds_since(m.last_seen)
        item["last_seen_seconds"] = elapsed
        item["stale"] = elapsed is not None and elapsed > _STALE_AFTER_SECONDS
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
    elapsed = _seconds_since(machine.last_seen)
    result["last_seen_seconds"] = elapsed
    result["stale"] = elapsed is not None and elapsed > _STALE_AFTER_SECONDS

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
# PATCH /api/machines/{id}  (mutate machine metadata — currently group_name)
# ---------------------------------------------------------------------------

class MachineUpdate(BaseModel):
    # Use Field(...) sentinel so we can distinguish "set to null" from "field omitted".
    group_name: Optional[str] = Field(default=None, max_length=64)


@router.patch("/machines/{machine_id}")
async def update_machine(
    machine_id: str,
    body: MachineUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update mutable machine fields. Currently supports group_name.

    Pass ``{"group_name": "training-east"}`` to assign, or ``{"group_name": ""}``
    / ``{"group_name": null}`` to ungroup.
    """
    machine = await session.get(Machine, machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    # _clean_group_name raises 400 on invalid characters.
    machine.group_name = _clean_group_name(body.group_name)
    await session.commit()
    await session.refresh(machine)

    result = machine.to_dict()
    elapsed = _seconds_since(machine.last_seen)
    result["last_seen_seconds"] = elapsed
    result["stale"] = elapsed is not None and elapsed > _STALE_AFTER_SECONDS
    if machine.latest_snapshot_id:
        snap = await session.get(Snapshot, machine.latest_snapshot_id)
        if snap:
            # Match GET /api/machines/{id} shape so callers (MachineDetail) can
            # safely setMachine(patchResponse) without losing diagnostics.
            result["latest_report"] = json.loads(snap.report_json)
            result["gpu_name"] = snap.gpu_name
            result["driver_version"] = snap.driver_version
            result["cuda_version"] = snap.cuda_version
            result["torch_version"] = snap.torch_version
    return result


# ---------------------------------------------------------------------------
# GET /api/groups
# ---------------------------------------------------------------------------

@router.get("/groups")
async def list_groups(session: AsyncSession = Depends(get_session)):
    """Return all distinct group names with member counts and status breakdown.

    NULL group_name values are aggregated under the synthetic "ungrouped" entry
    and listed last; named groups are sorted alphabetically.
    """
    pass_sum = func.sum(case((Machine.latest_status == "pass", 1), else_=0))
    warn_sum = func.sum(case((Machine.latest_status == "warning", 1), else_=0))
    fail_sum = func.sum(case((Machine.latest_status == "fail", 1), else_=0))

    query = (
        select(
            Machine.group_name,
            func.count(Machine.id),
            pass_sum,
            warn_sum,
            fail_sum,
        )
        .group_by(Machine.group_name)
    )
    result = await session.execute(query)

    named: list[dict] = []
    ungrouped: Optional[dict] = None
    for group_name, count, n_pass, n_warn, n_fail in result.all():
        entry = {
            "name": group_name if group_name else _UNGROUPED_LABEL,
            "machine_count": int(count or 0),
            "status_breakdown": {
                "pass": int(n_pass or 0),
                "warning": int(n_warn or 0),
                "fail": int(n_fail or 0),
            },
        }
        if group_name:
            named.append(entry)
        else:
            ungrouped = entry

    named.sort(key=lambda g: g["name"].lower())
    if ungrouped is not None:
        named.append(ungrouped)
    return named


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


# ---------------------------------------------------------------------------
# Command queue endpoints
# ---------------------------------------------------------------------------

class QueueCommandRequest(BaseModel):
    command: str


class CommandResultRequest(BaseModel):
    output: str
    exit_code: int


@router.post("/machines/{machine_id}/commands")
async def queue_command(
    machine_id: str,
    body: QueueCommandRequest,
    session: AsyncSession = Depends(get_session),
):
    """Queue a remediation command to be executed on the next machine check-in."""
    machine = await session.get(Machine, machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    # Only allow env-doctor commands for safety
    allowed_prefixes = ("env-doctor ", "doctor ")
    if not any(body.command.strip().startswith(p) for p in allowed_prefixes):
        raise HTTPException(status_code=400, detail="Only env-doctor commands may be queued")

    cmd = Command(machine_id=machine_id, command=body.command.strip(), status="pending")
    session.add(cmd)
    await session.commit()
    await session.refresh(cmd)
    return cmd.to_dict()


@router.post("/machines/{machine_id}/commands/{cmd_id}/result")
async def post_command_result(
    machine_id: str,
    cmd_id: int,
    body: CommandResultRequest,
    session: AsyncSession = Depends(get_session),
):
    """Receive execution result from a machine after running a queued command."""
    cmd = await session.get(Command, cmd_id)
    if not cmd or cmd.machine_id != machine_id:
        raise HTTPException(status_code=404, detail="Command not found")

    cmd.output = body.output
    cmd.exit_code = body.exit_code
    cmd.status = "done" if body.exit_code == 0 else "failed"
    cmd.executed_at = datetime.now(timezone.utc)
    await session.commit()
    return cmd.to_dict()


@router.get("/machines/{machine_id}/commands")
async def list_commands(
    machine_id: str,
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List recent commands for a machine (for dashboard polling)."""
    machine = await session.get(Machine, machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    query = (
        select(Command)
        .where(Command.machine_id == machine_id)
        .order_by(Command.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(query)
    return [c.to_dict() for c in result.scalars().all()]


# ---------------------------------------------------------------------------
# GET /api/commands  (cross-fleet activity log)
# ---------------------------------------------------------------------------

_VALID_COMMAND_STATUSES = {"pending", "running", "done", "failed"}


def _parse_iso(value: str) -> datetime:
    # datetime.fromisoformat in 3.11+ accepts trailing "Z"; older Pythons need a swap.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@router.get("/commands")
async def list_command_activity(
    status: Optional[str] = Query(None, description="pending | running | done | failed"),
    machine_id: Optional[str] = Query(None),
    since: Optional[str] = Query(None, description="ISO-8601 timestamp lower bound on created_at"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """Cross-fleet command activity log with hostname joined in."""
    if status is not None and status not in _VALID_COMMAND_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {sorted(_VALID_COMMAND_STATUSES)}",
        )

    since_dt: Optional[datetime] = None
    if since:
        try:
            since_dt = _parse_iso(since)
        except ValueError:
            raise HTTPException(status_code=400, detail="since must be an ISO-8601 timestamp")

    query = (
        select(Command, Machine.hostname)
        .join(Machine, Command.machine_id == Machine.id, isouter=True)
        .order_by(Command.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status:
        query = query.where(Command.status == status)
    if machine_id:
        query = query.where(Command.machine_id == machine_id)
    if since_dt is not None:
        query = query.where(Command.created_at >= since_dt)

    result = await session.execute(query)
    rows = result.all()

    output = []
    for cmd, hostname in rows:
        item = cmd.to_dict()
        item["hostname"] = hostname
        if cmd.created_at and cmd.executed_at:
            created = cmd.created_at if cmd.created_at.tzinfo else cmd.created_at.replace(tzinfo=timezone.utc)
            executed = cmd.executed_at if cmd.executed_at.tzinfo else cmd.executed_at.replace(tzinfo=timezone.utc)
            item["duration_seconds"] = (executed - created).total_seconds()
        else:
            item["duration_seconds"] = None
        output.append(item)
    return output

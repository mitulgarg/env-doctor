"""Render an env-doctor check result as self-contained HTML.

The input is the structured dict produced by ``cli._build_check_output``.
The output is a single HTML fragment with all styling inlined in a scoped
``<div>`` — no external stylesheets, no JavaScript — so it renders identically
in Jupyter, JupyterLab, VS Code, Colab, and saved ``.html`` files.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional

_LOGO_SVG = (
    Path(__file__).parent / "logo.svg"
).read_text(encoding="utf-8").strip().replace("\n", "").replace("  ", " ").replace(
    "<svg ", '<svg style="vertical-align:middle;flex-shrink:0;" ', 1
)

# --- Semantic status tokens ------------------------------------------------
# Per-check statuses come from core.detector.Status values; the overall status
# from determine_overall_status uses pass/warning/fail.
_STATUS_COLORS = {
    "success": ("#16a34a", "OK"),
    "warning": ("#d97706", "WARN"),
    "error": ("#dc2626", "ERROR"),
    "not_found": ("#64748b", "ABSENT"),
    # overall-status aliases
    "pass": ("#16a34a", "HEALTHY"),
    "fail": ("#dc2626", "ISSUES FOUND"),
    "unknown": ("#64748b", "UNKNOWN"),
    "compatible": ("#16a34a", "OK"),
    "mismatch": ("#dc2626", "MISMATCH"),
}

_BORDER = "#1f2937"
_SURFACE = "#0f172a"
_SURFACE_RAISED = "#1e293b"
_TEXT = "#e2e8f0"
_TEXT_MUTED = "#94a3b8"
_ACCENT = "#38bdf8"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _badge(status: str) -> str:
    color, label = _STATUS_COLORS.get(status, _STATUS_COLORS["unknown"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'background:{color};color:#fff;font-size:11px;font-weight:700;'
        f'letter-spacing:.04em;vertical-align:middle;white-space:nowrap;">'
        f"{_esc(label)}</span>"
    )


def _fmt_value(value: Any) -> str:
    """Render a metadata value as a compact, readable string."""
    if isinstance(value, dict):
        return ", ".join(f"{_esc(k)}={_fmt_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value) if value else "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _esc(value)


def _list_block(title: str, items, color: str) -> str:
    if not items:
        return ""
    rows = "".join(
        f'<li style="margin:2px 0;">{_esc(item)}</li>' for item in items
    )
    return (
        f'<div style="margin-top:8px;"><div style="color:{color};font-size:12px;'
        f'font-weight:600;margin-bottom:2px;">{_esc(title)}</div>'
        f'<ul style="margin:0;padding-left:18px;color:{_TEXT};font-size:13px;'
        f'line-height:1.5;">{rows}</ul></div>'
    )


def _metadata_block(metadata: Dict[str, Any]) -> str:
    visible = {k: v for k, v in (metadata or {}).items() if k != "detection_method"}
    if not visible:
        return ""
    rows = "".join(
        f'<tr><td style="color:{_TEXT_MUTED};padding:2px 12px 2px 0;'
        f'white-space:nowrap;vertical-align:top;">{_esc(k.replace("_", " "))}</td>'
        f'<td style="color:{_TEXT};padding:2px 0;">{_fmt_value(v)}</td></tr>'
        for k, v in visible.items()
    )
    return (
        '<details style="margin-top:8px;"><summary style="cursor:pointer;'
        f'color:{_ACCENT};font-size:12px;outline:none;">Details</summary>'
        f'<table style="margin-top:6px;border-collapse:collapse;font-size:13px;">'
        f"{rows}</table></details>"
    )


def _section(title: str, status: str, version: Optional[str],
             path: Optional[str], metadata: Optional[dict],
             issues, recommendations) -> str:
    head_meta = ""
    if version:
        head_meta += (
            f'<span style="color:{_ACCENT};font-family:ui-monospace,'
            f'SFMono-Regular,Menlo,monospace;font-size:13px;">{_esc(version)}</span>'
        )
    if path:
        head_meta += (
            f'<div style="color:{_TEXT_MUTED};font-size:12px;font-family:'
            f'ui-monospace,Menlo,monospace;margin-top:2px;word-break:break-all;">'
            f"{_esc(path)}</div>"
        )
    return (
        f'<div style="background:{_SURFACE_RAISED};border:1px solid {_BORDER};'
        'border-radius:10px;padding:14px 16px;margin:10px 0;">'
        '<div style="display:flex;align-items:center;gap:10px;'
        'justify-content:space-between;">'
        f'<span style="font-weight:600;font-size:15px;color:{_TEXT};">'
        f"{_esc(title)}</span>{_badge(status)}</div>"
        f'<div style="margin-top:4px;">{head_meta}</div>'
        f'{_list_block("Issues", issues, "#fca5a5")}'
        f'{_list_block("Recommendations", recommendations, "#7dd3fc")}'
        f"{_metadata_block(metadata)}"
        "</div>"
    )


def _render_check(title: str, check: Optional[dict]) -> str:
    if not check:
        return ""
    return _section(
        title=title,
        status=check.get("status", "unknown"),
        version=check.get("version"),
        path=check.get("path"),
        metadata=check.get("metadata"),
        issues=check.get("issues"),
        recommendations=check.get("recommendations"),
    )


def _render_compute_compat(info: Optional[dict]) -> str:
    if not info:
        return ""
    status = info.get("status", "unknown")
    gpu = info.get("gpu_name")
    arch = info.get("arch_name")
    sm = info.get("sm")
    version = f"{gpu} ({arch} {sm})" if gpu and arch and sm else gpu
    issues = []
    recs = []
    if info.get("message"):
        (issues if status == "mismatch" else recs).append(info["message"])
    if info.get("nightly_url"):
        recs.append(f"PyTorch nightly: {info['nightly_url']}")
    return _section(
        title="Compute Capability",
        status=status,
        version=version,
        path=None,
        metadata={"arch_list": info.get("arch_list"),
                  "cuda_available": info.get("cuda_available")},
        issues=issues,
        recommendations=recs,
    )


def format_result_html(output: Dict[str, Any]) -> str:
    """Build a self-contained HTML fragment from a check-output dict.

    Args:
        output: The dict produced by ``cli._build_check_output`` (keys:
            ``machine``, ``status``, ``timestamp``, ``summary``, ``checks``).

    Returns:
        An HTML string safe to embed directly (all values are escaped).
    """
    machine = output.get("machine", {})
    summary = output.get("summary", {})
    checks = output.get("checks", {})
    overall = output.get("status", "unknown")

    machine_line = " · ".join(
        part for part in (
            _esc(machine.get("hostname")) if machine.get("hostname") else "",
            _esc(machine.get("platform")) if machine.get("platform") else "",
            f"Python {_esc(machine.get('python_version'))}"
            if machine.get("python_version") else "",
        ) if part
    )

    issues_count = summary.get("issues_count", 0)
    issues_text = (
        f"{issues_count} issue{'s' if issues_count != 1 else ''} detected"
        if issues_count else "No issues detected"
    )

    sections = [
        _render_check("WSL2 Environment", checks.get("wsl2")),
        _render_check("NVIDIA Driver", checks.get("driver")),
        _render_check("CUDA Toolkit", checks.get("cuda")),
        _render_check("cuDNN", checks.get("cudnn")),
    ]
    for lib, lib_check in (checks.get("libraries") or {}).items():
        sections.append(_render_check(lib, lib_check))
    sections.append(_render_check("Python Compatibility", checks.get("python_compat")))
    sections.append(_render_compute_compat(checks.get("compute_compatibility")))
    body = "".join(s for s in sections if s)

    return (
        f'<div class="env-doctor-report" style="font-family:ui-sans-serif,'
        f'system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:{_SURFACE};'
        f'color:{_TEXT};border:1px solid {_BORDER};border-radius:14px;'
        'padding:20px 22px;max-width:760px;line-height:1.45;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        f'gap:12px;border-bottom:1px solid {_BORDER};padding-bottom:14px;">'
        '<div><div style="display:flex;align-items:center;gap:8px;'
        'font-size:18px;font-weight:700;letter-spacing:.02em;">'
        f'{_LOGO_SVG}<span>env-doctor</span></div>'
        f'<div style="color:{_TEXT_MUTED};font-size:12px;margin-top:2px;">'
        f"{machine_line}</div></div>"
        f'<div style="text-align:right;">{_badge(overall)}'
        f'<div style="color:{_TEXT_MUTED};font-size:12px;margin-top:6px;">'
        f"{_esc(issues_text)}</div></div></div>"
        f"{body}"
        f'<div style="color:{_TEXT_MUTED};font-size:11px;margin-top:14px;'
        f'text-align:right;">{_esc(output.get("timestamp", ""))}</div>'
        "</div>"
    )

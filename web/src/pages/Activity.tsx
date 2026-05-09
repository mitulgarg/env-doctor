import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getCommandActivity, getMachines } from "../api";
import type { CommandActivityRow, MachineListItem } from "../types";

const REFRESH_MS = 10_000;
const PAGE_LIMIT = 100;

type StatusFilter = "all" | "pending" | "running" | "done" | "failed";
type RangeFilter = "1h" | "24h" | "7d" | "all";

const RANGE_SECONDS: Record<Exclude<RangeFilter, "all">, number> = {
  "1h": 3600,
  "24h": 86_400,
  "7d": 604_800,
};

const cmdStatusColors: Record<string, { bg: string; fg: string }> = {
  pending: { bg: "rgba(139,149,163,0.18)", fg: "#b1bac4" },
  running: { bg: "rgba(88,166,255,0.18)", fg: "#58a6ff" },
  done: { bg: "rgba(35,134,54,0.22)", fg: "#3fb950" },
  failed: { bg: "rgba(218,54,51,0.22)", fg: "#f85149" },
};

function CmdStatusBadge({ status }: { status: string }) {
  const c = cmdStatusColors[status] ?? { bg: "rgba(255,255,255,0.12)", fg: "rgba(255,255,255,0.7)" };
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 10px",
      borderRadius: 12,
      fontSize: 11,
      fontWeight: 600,
      background: c.bg,
      color: c.fg,
      textTransform: "uppercase",
      letterSpacing: 0.5,
    }}>{status}</span>
  );
}

function timeAgo(iso: string | null): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.max(0, Math.floor(diff / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

const thS: React.CSSProperties = {
  padding: "10px 14px",
  textAlign: "left",
  fontSize: 11,
  fontWeight: 600,
  color: "rgba(255,255,255,0.35)",
  textTransform: "uppercase",
  letterSpacing: 0.8,
  borderBottom: "1px solid rgba(255,255,255,0.08)",
};
const tdS: React.CSSProperties = {
  padding: "12px 14px",
  fontSize: 13,
  color: "#e6edf3",
  borderBottom: "1px solid rgba(255,255,255,0.05)",
};

const filterBtnStyle = (active: boolean, color?: string): React.CSSProperties => ({
  padding: "5px 14px",
  border: `1px solid ${active ? (color ?? "#58a6ff") : "rgba(255,255,255,0.12)"}`,
  borderRadius: 20,
  background: active ? `${color ?? "#58a6ff"}22` : "transparent",
  color: active ? (color ?? "#58a6ff") : "rgba(255,255,255,0.5)",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: active ? 600 : 400,
});

export default function Activity() {
  const [rows, setRows] = useState<CommandActivityRow[]>([]);
  const [machines, setMachines] = useState<MachineListItem[]>([]);
  const [status, setStatus] = useState<StatusFilter>("all");
  const [machineId, setMachineId] = useState<string>("");
  const [range, setRange] = useState<RangeFilter>("24h");
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Machines list for the filter dropdown — load once.
  useEffect(() => {
    getMachines().then(setMachines).catch(() => {});
  }, []);

  // Build the `since` ISO string from the current range. Recompute on every load so
  // the rolling window stays accurate as time passes.
  const sinceFor = (r: RangeFilter): string | undefined => {
    if (r === "all") return undefined;
    return new Date(Date.now() - RANGE_SECONDS[r] * 1000).toISOString();
  };

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      getCommandActivity({
        status: status === "all" ? undefined : status,
        machine_id: machineId || undefined,
        since: sinceFor(range),
        limit: PAGE_LIMIT,
      })
        .then(data => {
          if (cancelled) return;
          setRows(data);
          setError(null);
        })
        .catch(err => {
          if (cancelled) return;
          setError(err?.message ?? "Failed to load");
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [status, machineId, range]);

  const counts = useMemo(() => ({
    all: rows.length,
    pending: rows.filter(r => r.status === "pending").length,
    running: rows.filter(r => r.status === "running").length,
    done: rows.filter(r => r.status === "done").length,
    failed: rows.filter(r => r.status === "failed").length,
  }), [rows]);

  return (
    <div style={{ flex: 1, overflow: "auto", padding: 28 }}>
      <h2 style={{ margin: "0 0 6px", fontSize: 20, fontWeight: 700, color: "#e6edf3" }}>
        Command Activity
      </h2>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", marginBottom: 20 }}>
        Cross-fleet log of remediation commands · refreshes every {REFRESH_MS / 1000}s
      </div>

      {/* Filters */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginRight: 4 }}>Status:</span>
        <button style={filterBtnStyle(status === "all")} onClick={() => setStatus("all")}>
          All ({counts.all})
        </button>
        <button style={filterBtnStyle(status === "pending", "#b1bac4")} onClick={() => setStatus("pending")}>
          Pending
        </button>
        <button style={filterBtnStyle(status === "running", "#58a6ff")} onClick={() => setStatus("running")}>
          Running
        </button>
        <button style={filterBtnStyle(status === "done", "#3fb950")} onClick={() => setStatus("done")}>
          Done
        </button>
        <button style={filterBtnStyle(status === "failed", "#f85149")} onClick={() => setStatus("failed")}>
          Failed
        </button>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)" }}>Machine:</span>
        <select
          value={machineId}
          onChange={e => setMachineId(e.target.value)}
          style={{
            padding: "6px 10px",
            background: "#0d1117",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            color: "#e6edf3",
            fontSize: 13,
            minWidth: 200,
          }}
        >
          <option value="">All machines</option>
          {machines.map(m => (
            <option key={m.id} value={m.id}>{m.hostname}</option>
          ))}
        </select>

        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginLeft: 12 }}>Range:</span>
        {(["1h", "24h", "7d", "all"] as RangeFilter[]).map(r => (
          <button key={r} style={filterBtnStyle(range === r)} onClick={() => setRange(r)}>
            {r === "all" ? "All time" : `Last ${r}`}
          </button>
        ))}

        {(status !== "all" || machineId || range !== "24h") && (
          <button
            onClick={() => { setStatus("all"); setMachineId(""); setRange("24h"); }}
            style={{
              padding: "6px 12px",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 8,
              color: "rgba(255,255,255,0.5)",
              fontSize: 12,
              cursor: "pointer",
              marginLeft: 8,
            }}
          >
            Reset
          </button>
        )}
      </div>

      {error && (
        <div style={{
          padding: 12,
          marginBottom: 16,
          border: "1px solid rgba(218,54,51,0.4)",
          background: "rgba(218,54,51,0.08)",
          borderRadius: 8,
          color: "#f85149",
          fontSize: 13,
        }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "rgba(255,255,255,0.3)" }}>Loading…</div>
      ) : rows.length === 0 ? (
        <div style={{
          textAlign: "center",
          padding: 48,
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          background: "rgba(255,255,255,0.02)",
        }}>
          <div style={{ fontSize: 14, color: "rgba(255,255,255,0.5)" }}>No commands match these filters.</div>
        </div>
      ) : (
        <div style={{
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          overflow: "hidden",
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["Time", "Machine", "Command", "Status", "Exit", "Duration"].map(h => (
                  <th key={h} style={thS}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(r => {
                const isExpanded = expandedId === r.id;
                const hasOutput = !!r.output && r.output.trim().length > 0;
                return (
                  <>
                    <tr
                      key={r.id}
                      onClick={() => hasOutput && setExpandedId(isExpanded ? null : r.id)}
                      style={{
                        cursor: hasOutput ? "pointer" : "default",
                        background: isExpanded ? "rgba(255,255,255,0.04)" : undefined,
                      }}
                      onMouseEnter={e => { if (!isExpanded && hasOutput) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                      onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = ""; }}
                    >
                      <td style={{ ...tdS, color: "rgba(255,255,255,0.55)", fontSize: 12, whiteSpace: "nowrap" }}
                          title={r.created_at ?? ""}>
                        <span style={{
                          display: "inline-block",
                          width: 10,
                          color: hasOutput ? (isExpanded ? "#58a6ff" : "rgba(255,255,255,0.3)") : "transparent",
                          transition: "transform .15s",
                          transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                          marginRight: 6,
                        }}>▶</span>
                        {timeAgo(r.created_at)}
                      </td>
                      <td style={tdS}>
                        {r.hostname ? (
                          <Link
                            to={`/machines/${r.machine_id}`}
                            onClick={e => e.stopPropagation()}
                            style={{ color: "#58a6ff", textDecoration: "none" }}
                          >
                            {r.hostname}
                          </Link>
                        ) : (
                          <span style={{ color: "rgba(255,255,255,0.3)" }}>{r.machine_id.slice(0, 8)}…</span>
                        )}
                      </td>
                      <td style={{
                        ...tdS,
                        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                        fontSize: 12,
                        color: "rgba(230,237,243,0.9)",
                        maxWidth: 420,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }} title={r.command}>
                        {r.command}
                      </td>
                      <td style={tdS}><CmdStatusBadge status={r.status} /></td>
                      <td style={{
                        ...tdS,
                        fontWeight: 600,
                        color: r.exit_code == null
                          ? "rgba(255,255,255,0.3)"
                          : r.exit_code === 0 ? "#3fb950" : "#f85149",
                      }}>
                        {r.exit_code ?? "—"}
                      </td>
                      <td style={{ ...tdS, color: "rgba(255,255,255,0.55)", fontSize: 12 }}>
                        {formatDuration(r.duration_seconds)}
                      </td>
                    </tr>
                    {isExpanded && hasOutput && (
                      <tr key={`${r.id}-out`}>
                        <td colSpan={6} style={{ padding: 0, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                          <pre style={{
                            margin: 0,
                            padding: "16px 20px",
                            background: "rgba(13,17,23,0.8)",
                            color: "rgba(230,237,243,0.85)",
                            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                            fontSize: 12,
                            lineHeight: 1.5,
                            whiteSpace: "pre-wrap",
                            wordBreak: "break-word",
                            maxHeight: 400,
                            overflow: "auto",
                          }}>
                            {r.output}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {rows.length === PAGE_LIMIT && (
        <div style={{ textAlign: "center", padding: 16, fontSize: 12, color: "rgba(255,255,255,0.4)" }}>
          Showing latest {PAGE_LIMIT}. Tighten the time range or pick a machine to narrow further.
        </div>
      )}
    </div>
  );
}

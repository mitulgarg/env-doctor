import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getMachines, getMachine } from "../api";
import type { MachineDetail, MachineListItem } from "../types";
import StatusBadge from "../components/StatusBadge";
import PieChart from "../components/PieChart";
import HealthGauge from "../components/HealthGauge";
import CommandBlock from "../components/CommandBlock";
import DiagnosticCard from "../components/DiagnosticCard";

const REFRESH_MS = 30_000;

/* ─── Issue → env-doctor command mapping ─── */
function recommendCommands(_machineId: string, detail: MachineDetail): { label: string; command: string }[] {
  const report = detail.latest_report;
  if (!report) return [];
  const cmds: { label: string; command: string }[] = [];
  const checks = report.checks;

  const libs = checks.libraries ?? {};
  for (const [lib, result] of Object.entries(libs)) {
    if (result.status === "error" || result.status === "warning") {
      cmds.push({
        label: `Fix ${lib}`,
        command: `env-doctor install ${lib} --execute`,
      });
    } else if (result.status === "not_found") {
      cmds.push({
        label: `Install ${lib}`,
        command: `env-doctor install ${lib} --execute`,
      });
    }
  }

  if (checks.driver?.status === "error" || checks.driver?.status === "not_found") {
    cmds.push({ label: "Get driver info", command: "env-doctor cuda-info" });
  }

  if (checks.cuda?.status === "not_found" || checks.cuda?.status === "error") {
    cmds.push({ label: "CUDA toolkit info", command: "env-doctor cuda-info" });
  }

  // Always offer a re-check
  cmds.push({ label: "Re-run diagnostics", command: `env-doctor check --json` });

  return cmds;
}

function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
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

export default function FleetOverview() {
  const [machines, setMachines] = useState<MachineListItem[]>([]);
  const [filter, setFilter] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [gpuFilter, setGpuFilter] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detailCache, setDetailCache] = useState<Map<string, MachineDetail>>(new Map());

  const load = () => {
    // We only push status to the server; search/gpu/stale filters are client-side
    // so the user can flip filters without re-fetching.
    const serverStatus = filter && filter !== "stale" ? filter : undefined;
    getMachines(serverStatus)
      .then(setMachines)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [filter]);

  const handleRowClick = async (id: string) => {
    if (expandedId === id) { setExpandedId(null); return; }
    setExpandedId(id);
    if (!detailCache.has(id)) {
      try {
        const detail = await getMachine(id);
        setDetailCache(prev => new Map(prev).set(id, detail));
      } catch {
        // ignore
      }
    }
  };

  const allMachines = machines;
  const counts = {
    pass: allMachines.filter(m => m.latest_status === "pass").length,
    warning: allMachines.filter(m => m.latest_status === "warning").length,
    fail: allMachines.filter(m => m.latest_status === "fail").length,
    stale: allMachines.filter(m => m.stale).length,
  };

  const gpuOptions = Array.from(
    new Set(allMachines.map(m => m.gpu_name).filter((v): v is string => !!v))
  ).sort();

  const matchesSearch = (m: MachineListItem) => {
    const q = search.trim().toLowerCase();
    if (!q) return true;
    return m.hostname.toLowerCase().includes(q);
  };
  const matchesGpu = (m: MachineListItem) => !gpuFilter || m.gpu_name === gpuFilter;

  // Issues-focused: show all that aren't passing or are stale (unless a filter is active)
  const baseMachines = filter === "stale"
    ? machines.filter(m => m.stale)
    : filter
      ? machines
      : machines.filter(m => m.latest_status !== "pass" || m.stale);

  const issuesMachines = baseMachines.filter(m => matchesSearch(m) && matchesGpu(m));

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

  return (
    <div style={{ flex: 1, overflow: "auto", padding: 28 }}>

      {/* Header */}
      <h2 style={{ margin: "0 0 24px", fontSize: 20, fontWeight: 700, color: "#e6edf3" }}>
        Fleet Health
      </h2>

      {/* Top section: pie chart + counts */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 32,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 12,
        padding: "20px 28px",
        marginBottom: 28,
      }}>
        <PieChart pass={counts.pass} warning={counts.warning} fail={counts.fail} />
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {[
            { label: "Healthy", count: counts.pass, color: "#238636" },
            { label: "Warning", count: counts.warning, color: "#d29922" },
            { label: "Failed", count: counts.fail, color: "#da3633" },
            { label: "Stale", count: counts.stale, color: "#8b95a3" },
          ].map(item => (
            <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: item.color, boxShadow: `0 0 6px ${item.color}` }} />
              <span style={{ fontSize: 14, color: "rgba(255,255,255,0.6)", minWidth: 60 }}>{item.label}</span>
              <span style={{ fontSize: 20, fontWeight: 700, color: "#e6edf3" }}>{item.count}</span>
            </div>
          ))}
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginTop: 4 }}>
            {allMachines.length} total · refreshes every 30s
          </div>
        </div>
      </div>

      {/* Search + GPU filter */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search hostname…"
          style={{
            flex: 1,
            maxWidth: 360,
            padding: "8px 12px",
            background: "#0d1117",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            color: "#e6edf3",
            fontSize: 13,
          }}
        />
        <select
          value={gpuFilter}
          onChange={e => setGpuFilter(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#0d1117",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            color: "#e6edf3",
            fontSize: 13,
            minWidth: 180,
          }}
        >
          <option value="">All GPUs</option>
          {gpuOptions.map(g => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
        {(search || gpuFilter) && (
          <button
            onClick={() => { setSearch(""); setGpuFilter(""); }}
            style={{
              padding: "8px 12px",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 8,
              color: "rgba(255,255,255,0.5)",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Status filters */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginRight: 4 }}>Filter:</span>
        <button style={filterBtnStyle(filter === null)} onClick={() => setFilter(null)}>
          Issues only
        </button>
        <button style={filterBtnStyle(filter === "warning", "#d29922")} onClick={() => setFilter(filter === "warning" ? null : "warning")}>
          Warning
        </button>
        <button style={filterBtnStyle(filter === "fail", "#da3633")} onClick={() => setFilter(filter === "fail" ? null : "fail")}>
          Failed
        </button>
        <button style={filterBtnStyle(filter === "pass", "#238636")} onClick={() => setFilter(filter === "pass" ? null : "pass")}>
          Healthy
        </button>
        <button style={filterBtnStyle(filter === "stale", "#8b95a3")} onClick={() => setFilter(filter === "stale" ? null : "stale")}>
          Stale
        </button>
      </div>

      {/* Issues section title */}
      <div style={{ fontSize: 14, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginBottom: 12 }}>
        {filter ? `Showing: ${filter}` : "Critical Issues"}
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "rgba(255,255,255,0.3)" }}>Loading…</div>
      ) : issuesMachines.length === 0 ? (
        <div style={{
          textAlign: "center",
          padding: 48,
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          background: "rgba(35,134,54,0.05)",
        }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>✓</div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#238636" }}>All systems healthy</div>
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", marginTop: 8 }}>
            No machines are reporting issues right now.
          </div>
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
                {["Machine", "GPU", "Status", "Issues", "Last Seen"].map(h => (
                  <th key={h} style={thS}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {issuesMachines.map(m => {
                const isExpanded = expandedId === m.id;
                const detail = detailCache.get(m.id);
                const issueCount = detail?.latest_report?.summary?.issues_count ?? "—";
                const score = detail ? Math.max(0, 100 - (detail.latest_report?.summary?.issues_count ?? 0) * 15) : 0;
                const recs = detail ? recommendCommands(m.id, detail) : [];

                return (
                  <>
                    <tr
                      key={m.id}
                      onClick={() => handleRowClick(m.id)}
                      style={{ cursor: "pointer", background: isExpanded ? "rgba(255,255,255,0.04)" : undefined }}
                      onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                      onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = ""; }}
                    >
                      <td style={tdS}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{
                            fontSize: 11,
                            color: isExpanded ? "#58a6ff" : "rgba(255,255,255,0.3)",
                            transition: "transform .15s",
                            display: "inline-block",
                            transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                          }}>▶</span>
                          <Link
                            to={`/machines/${m.id}`}
                            onClick={e => e.stopPropagation()}
                            style={{
                              color: m.stale ? "rgba(230,237,243,0.55)" : "#e6edf3",
                              textDecoration: "none",
                              fontWeight: 600,
                            }}
                            onMouseEnter={e => (e.currentTarget.style.color = "#58a6ff")}
                            onMouseLeave={e => (e.currentTarget.style.color = m.stale ? "rgba(230,237,243,0.55)" : "#e6edf3")}
                            title="Open full details page"
                          >
                            {m.hostname}
                          </Link>
                          {m.stale && (
                            <span
                              title="No report received in over an hour"
                              style={{
                                fontSize: 10,
                                fontWeight: 600,
                                padding: "2px 7px",
                                borderRadius: 10,
                                background: "rgba(139,149,163,0.18)",
                                color: "#b1bac4",
                                border: "1px solid rgba(139,149,163,0.4)",
                                letterSpacing: 0.4,
                              }}
                            >
                              STALE
                            </span>
                          )}
                        </div>
                      </td>
                      <td style={{ ...tdS, color: "rgba(255,255,255,0.7)" }}>{m.gpu_name ?? "—"}</td>
                      <td style={tdS}><StatusBadge status={m.latest_status} /></td>
                      <td style={{ ...tdS, fontWeight: 600, color: m.latest_status === "fail" ? "#da3633" : "#d29922" }}>
                        {issueCount}
                      </td>
                      <td style={{ ...tdS, color: "rgba(255,255,255,0.4)", fontSize: 12 }}>{timeAgo(m.last_seen)}</td>
                    </tr>

                    {isExpanded && (
                      <tr key={`${m.id}-expand`}>
                        <td colSpan={5} style={{ padding: 0, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                          <div style={{
                            background: "rgba(13,17,23,0.8)",
                            borderTop: "1px solid rgba(255,255,255,0.06)",
                            padding: 24,
                          }}>
                            {!detail ? (
                              <div style={{ color: "rgba(255,255,255,0.3)", textAlign: "center", padding: 20 }}>
                                Loading machine details…
                              </div>
                            ) : (
                              <>
                                {/* Top action bar — primary navigation to MachineDetail */}
                                <div style={{
                                  display: "flex",
                                  justifyContent: "space-between",
                                  alignItems: "center",
                                  marginBottom: 18,
                                  paddingBottom: 14,
                                  borderBottom: "1px solid rgba(255,255,255,0.06)",
                                }}>
                                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>
                                    Quick view · use full details for command runner, history, and group editor
                                  </div>
                                  <Link
                                    to={`/machines/${m.id}`}
                                    onClick={e => e.stopPropagation()}
                                    style={{
                                      padding: "8px 14px",
                                      background: "#1f6feb",
                                      color: "#fff",
                                      borderRadius: 6,
                                      fontSize: 13,
                                      fontWeight: 600,
                                      textDecoration: "none",
                                      display: "inline-flex",
                                      alignItems: "center",
                                      gap: 6,
                                      transition: "background .15s",
                                    }}
                                    onMouseEnter={e => (e.currentTarget.style.background = "#388bfd")}
                                    onMouseLeave={e => (e.currentTarget.style.background = "#1f6feb")}
                                  >
                                    Open full details →
                                  </Link>
                                </div>

                                {/* Gauge + Commands row */}
                                <div style={{ display: "flex", gap: 24, marginBottom: 24, alignItems: "flex-start" }}>
                                  {/* Left: Gauge */}
                                  <div style={{
                                    background: "rgba(255,255,255,0.03)",
                                    border: "1px solid rgba(255,255,255,0.07)",
                                    borderRadius: 10,
                                    padding: "20px 28px",
                                    display: "flex",
                                    flexDirection: "column",
                                    alignItems: "center",
                                    gap: 12,
                                    flexShrink: 0,
                                  }}>
                                    <HealthGauge score={score} />
                                    <div style={{ textAlign: "center" }}>
                                      <div style={{ fontSize: 14, fontWeight: 700, color: "#e6edf3" }}>{m.hostname}</div>
                                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>
                                        {m.gpu_name ?? "No GPU"} · {m.cuda_version ? `CUDA ${m.cuda_version}` : "No CUDA"}
                                      </div>
                                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 2 }}>
                                        Driver {m.driver_version ?? "—"}
                                      </div>
                                    </div>
                                  </div>

                                  {/* Right: Recommended actions */}
                                  <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginBottom: 12 }}>
                                      Recommended Actions
                                    </div>
                                    {recs.length === 0 ? (
                                      <div style={{ fontSize: 13, color: "rgba(255,255,255,0.3)" }}>
                                        No actionable commands available.
                                      </div>
                                    ) : (
                                      recs.map((rec, i) => (
                                        <CommandBlock
                                          key={i}
                                          machineId={m.id}
                                          command={rec.command}
                                          label={rec.label}
                                        />
                                      ))
                                    )}
                                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginTop: 8 }}>
                                      Commands run on the machine on next check-in via <code>env-doctor check --report-to</code>
                                    </div>
                                  </div>
                                </div>

                                {/* Diagnostics */}
                                {detail.latest_report?.checks && (
                                  <div>
                                    <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginBottom: 12 }}>
                                      Diagnostics
                                    </div>
                                    <DiagnosticCard title="GPU / Driver" result={detail.latest_report.checks.driver} />
                                    <DiagnosticCard title="CUDA Toolkit" result={detail.latest_report.checks.cuda} />
                                    <DiagnosticCard title="cuDNN" result={detail.latest_report.checks.cudnn} />
                                    <DiagnosticCard title="Python Compat" result={detail.latest_report.checks.python_compat} />
                                    {Object.entries(detail.latest_report.checks.libraries ?? {}).map(([lib, res]) => (
                                      <DiagnosticCard key={lib} title={`Library: ${lib}`} result={res} />
                                    ))}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
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
    </div>
  );
}

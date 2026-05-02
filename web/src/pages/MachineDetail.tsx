import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getMachine, getMachineHistory } from "../api";
import type { MachineDetail, SnapshotSummary } from "../types";
import StatusBadge from "../components/StatusBadge";
import DiagnosticCard from "../components/DiagnosticCard";
import CustomCommandBox from "../components/CustomCommandBox";

function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const card: React.CSSProperties = {
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 8,
};

export default function MachineDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [machine, setMachine] = useState<MachineDetail | null>(null);
  const [history, setHistory] = useState<SnapshotSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    if (!id) return;
    getMachine(id).then(setMachine).catch((e) => setError(e.message));
    getMachineHistory(id).then(setHistory).catch(console.error);
  };

  useEffect(() => { refresh(); }, [id]);

  if (error) {
    return (
      <div style={{ flex: 1, overflow: "auto", padding: 24, textAlign: "center" }}>
        <p style={{ color: "#f85149" }}>Error: {error}</p>
        <Link to="/fleet" style={{ color: "#58a6ff" }}>Back to Fleet</Link>
      </div>
    );
  }

  if (!machine) {
    return <div style={{ flex: 1, overflow: "auto", padding: 24, textAlign: "center", color: "rgba(255,255,255,0.3)" }}>Loading…</div>;
  }

  const report = machine.latest_report;
  const checks = report?.checks;

  const thS: React.CSSProperties = {
    textAlign: "left",
    padding: "8px 12px",
    fontSize: 12,
    fontWeight: 600,
    color: "rgba(255,255,255,0.35)",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  };
  const tdS: React.CSSProperties = {
    padding: "8px 12px",
    fontSize: 13,
    color: "#e6edf3",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
  };

  return (
    <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
      {/* Breadcrumb */}
      <div style={{ marginBottom: 16, fontSize: 13 }}>
        <Link to="/fleet" style={{ color: "#58a6ff", textDecoration: "none" }}>Fleet</Link>
        <span style={{ margin: "0 8px", color: "rgba(255,255,255,0.2)" }}>/</span>
        <span style={{ color: "rgba(255,255,255,0.6)" }}>{machine.hostname}</span>
      </div>

      {/* Header */}
      <div style={{ ...card, padding: 20, marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ margin: "0 0 4px", fontSize: 22, color: "#e6edf3" }}>{machine.hostname}</h2>
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.45)" }}>
            {machine.platform} | Python {machine.python_version} | ID: {machine.id}
          </div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", marginTop: 4 }}>
            Last seen: {timeAgo(machine.last_seen)}
            {machine.stale && (
              <span style={{
                marginLeft: 8,
                fontSize: 10,
                fontWeight: 600,
                padding: "2px 7px",
                borderRadius: 10,
                background: "rgba(139,149,163,0.18)",
                color: "#b1bac4",
                border: "1px solid rgba(139,149,163,0.4)",
                letterSpacing: 0.4,
              }}>
                STALE
              </span>
            )}
          </div>
        </div>
        <StatusBadge status={machine.latest_status} />
      </div>

      {/* Run a command */}
      {id && <CustomCommandBox machineId={id} onComplete={refresh} />}

      {/* Summary row */}
      {report && (
        <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
          {[
            { label: "GPU", value: machine.gpu_name },
            { label: "Driver", value: machine.driver_version },
            { label: "CUDA", value: machine.cuda_version },
            { label: "PyTorch", value: machine.torch_version },
            { label: "Issues", value: String(report.summary.issues_count) },
          ].map((item) => (
            <div key={item.label} style={{ ...card, flex: 1, padding: 12, textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", textTransform: "uppercase" }}>{item.label}</div>
              <div style={{ fontSize: 14, fontWeight: 600, marginTop: 4, color: "#e6edf3" }}>{item.value ?? "—"}</div>
            </div>
          ))}
        </div>
      )}

      {/* Diagnostic Cards */}
      {checks && (
        <div>
          <h3 style={{ fontSize: 16, marginBottom: 12, color: "rgba(255,255,255,0.6)" }}>Diagnostics</h3>
          <DiagnosticCard title="GPU / Driver" result={checks.driver} />
          <DiagnosticCard title="CUDA Toolkit" result={checks.cuda} />
          <DiagnosticCard title="cuDNN" result={checks.cudnn} />
          <DiagnosticCard title="WSL2" result={checks.wsl2} />
          <DiagnosticCard title="Python Compatibility" result={checks.python_compat} />
          {checks.libraries && Object.entries(checks.libraries).map(([lib, result]) => (
            <DiagnosticCard key={lib} title={`Library: ${lib}`} result={result} />
          ))}
          {checks.compute_compatibility && (
            <div style={{ ...card, padding: 16, marginBottom: 12 }}>
              <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#e6edf3" }}>Compute Compatibility</h3>
              <div style={{ fontSize: 13, color: "rgba(255,255,255,0.7)" }}>
                <div>GPU: <strong style={{ color: "#e6edf3" }}>{checks.compute_compatibility.gpu_name}</strong></div>
                <div>SM: {checks.compute_compatibility.sm ?? "—"} ({checks.compute_compatibility.arch_name ?? "—"})</div>
                <div style={{ marginTop: 4 }}>
                  Status: <StatusBadge status={checks.compute_compatibility.status === "compatible" ? "pass" : "fail"} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: 16, marginBottom: 12, color: "rgba(255,255,255,0.6)" }}>Report History</h3>
          <div style={{ ...card, overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Time", "Status", "GPU", "Driver", "CUDA", "Heartbeat"].map((h) => (
                    <th key={h} style={thS}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((snap) => (
                  <tr key={snap.id}>
                    <td style={tdS}>{new Date(snap.timestamp).toLocaleString()}</td>
                    <td style={tdS}><StatusBadge status={snap.status} /></td>
                    <td style={tdS}>{snap.gpu_name ?? "—"}</td>
                    <td style={tdS}>{snap.driver_version ?? "—"}</td>
                    <td style={tdS}>{snap.cuda_version ?? "—"}</td>
                    <td style={tdS}>{snap.is_heartbeat ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

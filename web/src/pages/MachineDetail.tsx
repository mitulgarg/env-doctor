import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getMachine, getMachineHistory } from "../api";
import type { MachineDetail, SnapshotSummary } from "../types";
import StatusBadge from "../components/StatusBadge";
import DiagnosticCard from "../components/DiagnosticCard";

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

export default function MachineDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [machine, setMachine] = useState<MachineDetail | null>(null);
  const [history, setHistory] = useState<SnapshotSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    getMachine(id).then(setMachine).catch((e) => setError(e.message));
    getMachineHistory(id).then(setHistory).catch(console.error);
  }, [id]);

  if (error) {
    return (
      <div style={{ padding: 40, textAlign: "center" }}>
        <p style={{ color: "#c92a2a" }}>Error: {error}</p>
        <Link to="/">Back to Fleet Overview</Link>
      </div>
    );
  }

  if (!machine) {
    return <div style={{ padding: 40, textAlign: "center", color: "#868e96" }}>Loading...</div>;
  }

  const report = machine.latest_report;
  const checks = report?.checks;

  return (
    <div>
      {/* Breadcrumb */}
      <div style={{ marginBottom: 16, fontSize: 13 }}>
        <Link to="/" style={{ color: "#228be6" }}>Fleet Overview</Link>
        <span style={{ margin: "0 8px", color: "#adb5bd" }}>/</span>
        <span>{machine.hostname}</span>
      </div>

      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "#fff",
          padding: 20,
          borderRadius: 8,
          border: "1px solid #dee2e6",
          marginBottom: 20,
        }}
      >
        <div>
          <h2 style={{ margin: "0 0 4px 0", fontSize: 22 }}>{machine.hostname}</h2>
          <div style={{ fontSize: 13, color: "#868e96" }}>
            {machine.platform} | Python {machine.python_version} | ID: {machine.id}
          </div>
          <div style={{ fontSize: 12, color: "#adb5bd", marginTop: 4 }}>
            Last seen: {timeAgo(machine.last_seen)}
          </div>
        </div>
        <StatusBadge status={machine.latest_status} />
      </div>

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
            <div
              key={item.label}
              style={{
                flex: 1,
                background: "#fff",
                border: "1px solid #dee2e6",
                borderRadius: 8,
                padding: 12,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 11, color: "#868e96", textTransform: "uppercase" }}>{item.label}</div>
              <div style={{ fontSize: 14, fontWeight: 600, marginTop: 4 }}>{item.value ?? "-"}</div>
            </div>
          ))}
        </div>
      )}

      {/* Diagnostic Cards */}
      {checks && (
        <div>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>Diagnostics</h3>
          <DiagnosticCard title="GPU / Driver" result={checks.driver} />
          <DiagnosticCard title="CUDA Toolkit" result={checks.cuda} />
          <DiagnosticCard title="cuDNN" result={checks.cudnn} />
          <DiagnosticCard title="WSL2" result={checks.wsl2} />
          <DiagnosticCard title="Python Compatibility" result={checks.python_compat} />

          {checks.libraries &&
            Object.entries(checks.libraries).map(([lib, result]) => (
              <DiagnosticCard key={lib} title={`Library: ${lib}`} result={result} />
            ))}

          {checks.compute_compatibility && (
            <div
              style={{
                border: "1px solid #dee2e6",
                borderRadius: 8,
                padding: 16,
                marginBottom: 12,
                background: "#fff",
              }}
            >
              <h3 style={{ margin: "0 0 8px 0", fontSize: 15 }}>Compute Compatibility</h3>
              <div style={{ fontSize: 13 }}>
                <div>GPU: <strong>{checks.compute_compatibility.gpu_name}</strong></div>
                <div>SM: {checks.compute_compatibility.sm ?? "-"} ({checks.compute_compatibility.arch_name ?? "-"})</div>
                <div>Status: <StatusBadge status={checks.compute_compatibility.status === "compatible" ? "pass" : "fail"} /></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>Report History</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", background: "#fff", borderRadius: 8 }}>
            <thead>
              <tr>
                {["Time", "Status", "GPU", "Driver", "CUDA", "Heartbeat"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: "left",
                      padding: "8px 12px",
                      fontSize: 12,
                      fontWeight: 600,
                      color: "#868e96",
                      borderBottom: "2px solid #dee2e6",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((snap) => (
                <tr key={snap.id}>
                  <td style={{ padding: "8px 12px", fontSize: 13, borderBottom: "1px solid #f1f3f5" }}>
                    {new Date(snap.timestamp).toLocaleString()}
                  </td>
                  <td style={{ padding: "8px 12px", borderBottom: "1px solid #f1f3f5" }}>
                    <StatusBadge status={snap.status} />
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: 13, borderBottom: "1px solid #f1f3f5" }}>
                    {snap.gpu_name ?? "-"}
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: 13, borderBottom: "1px solid #f1f3f5" }}>
                    {snap.driver_version ?? "-"}
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: 13, borderBottom: "1px solid #f1f3f5" }}>
                    {snap.cuda_version ?? "-"}
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: 13, borderBottom: "1px solid #f1f3f5" }}>
                    {snap.is_heartbeat ? "Yes" : "No"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

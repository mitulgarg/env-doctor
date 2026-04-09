import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { MachineListItem } from "../types";
import StatusBadge from "./StatusBadge";

type SortKey = keyof MachineListItem;

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

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  fontSize: 12,
  fontWeight: 600,
  color: "#868e96",
  textTransform: "uppercase",
  letterSpacing: 0.5,
  cursor: "pointer",
  userSelect: "none",
  borderBottom: "2px solid #dee2e6",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  fontSize: 13,
  borderBottom: "1px solid #f1f3f5",
};

interface Props {
  machines: MachineListItem[];
}

export default function MachineTable({ machines }: Props) {
  const navigate = useNavigate();
  const [sortKey, setSortKey] = useState<SortKey>("last_seen");
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sorted = [...machines].sort((a, b) => {
    const av = a[sortKey] ?? "";
    const bv = b[sortKey] ?? "";
    const cmp = String(av).localeCompare(String(bv));
    return sortAsc ? cmp : -cmp;
  });

  const cols: { key: SortKey; label: string }[] = [
    { key: "hostname", label: "Hostname" },
    { key: "latest_status", label: "Status" },
    { key: "gpu_name", label: "GPU" },
    { key: "driver_version", label: "Driver" },
    { key: "cuda_version", label: "CUDA" },
    { key: "torch_version", label: "PyTorch" },
    { key: "last_seen", label: "Last Seen" },
  ];

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", background: "#fff", borderRadius: 8 }}>
      <thead>
        <tr>
          {cols.map((c) => (
            <th key={c.key} style={thStyle} onClick={() => handleSort(c.key)}>
              {c.label} {sortKey === c.key ? (sortAsc ? "\u25B2" : "\u25BC") : ""}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((m) => (
          <tr
            key={m.id}
            onClick={() => navigate(`/machines/${m.id}`)}
            style={{ cursor: "pointer" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#f8f9fa")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "")}
          >
            <td style={tdStyle}><strong>{m.hostname}</strong></td>
            <td style={tdStyle}><StatusBadge status={m.latest_status} /></td>
            <td style={tdStyle}>{m.gpu_name ?? "-"}</td>
            <td style={tdStyle}>{m.driver_version ?? "-"}</td>
            <td style={tdStyle}>{m.cuda_version ?? "-"}</td>
            <td style={tdStyle}>{m.torch_version ?? "-"}</td>
            <td style={tdStyle}>{timeAgo(m.last_seen)}</td>
          </tr>
        ))}
        {sorted.length === 0 && (
          <tr>
            <td colSpan={cols.length} style={{ ...tdStyle, textAlign: "center", color: "#868e96", padding: 40 }}>
              No machines reporting yet. Run: <code>env-doctor check --report-to http://this-server:8765</code>
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

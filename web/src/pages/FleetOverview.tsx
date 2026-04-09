import { useEffect, useState } from "react";
import { getMachines } from "../api";
import type { MachineListItem } from "../types";
import MachineTable from "../components/MachineTable";
import StatusBadge from "../components/StatusBadge";

const REFRESH_INTERVAL = 30_000; // 30s

const filterBtnStyle = (active: boolean): React.CSSProperties => ({
  padding: "6px 14px",
  border: "1px solid " + (active ? "#228be6" : "#dee2e6"),
  borderRadius: 6,
  background: active ? "#e7f5ff" : "#fff",
  color: active ? "#1971c2" : "#495057",
  cursor: "pointer",
  fontSize: 13,
  fontWeight: active ? 600 : 400,
});

export default function FleetOverview() {
  const [machines, setMachines] = useState<MachineListItem[]>([]);
  const [filter, setFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    getMachines(filter ?? undefined)
      .then(setMachines)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_INTERVAL);
    return () => clearInterval(id);
  }, [filter]);

  const counts = {
    all: machines.length,
    pass: machines.filter((m) => m.latest_status === "pass").length,
    warning: machines.filter((m) => m.latest_status === "warning").length,
    fail: machines.filter((m) => m.latest_status === "fail").length,
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>Fleet Overview</h2>
        <div style={{ display: "flex", gap: 8 }}>
          {([null, "pass", "warning", "fail"] as const).map((f) => {
            const label = f ?? "all";
            const count = f ? counts[f] : counts.all;
            return (
              <button key={label} style={filterBtnStyle(filter === f)} onClick={() => setFilter(f)}>
                {label.charAt(0).toUpperCase() + label.slice(1)} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Summary cards */}
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        {(["pass", "warning", "fail"] as const).map((s) => (
          <div
            key={s}
            style={{
              flex: 1,
              padding: 16,
              borderRadius: 8,
              background: "#fff",
              border: "1px solid #dee2e6",
              textAlign: "center",
            }}
          >
            <StatusBadge status={s} />
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 8 }}>{counts[s]}</div>
            <div style={{ fontSize: 12, color: "#868e96" }}>machines</div>
          </div>
        ))}
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "#868e96" }}>Loading...</div>
      ) : (
        <MachineTable machines={filter ? machines : machines} />
      )}
    </div>
  );
}

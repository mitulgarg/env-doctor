import type { CommandRecord, MachineListItem, MachineDetail, SnapshotSummary } from "./types";

const BASE = "/api";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

export function getMachines(status?: string): Promise<MachineListItem[]> {
  const params = status ? `?status=${status}` : "";
  return fetchJson(`${BASE}/machines${params}`);
}

export function getMachine(id: string): Promise<MachineDetail> {
  return fetchJson(`${BASE}/machines/${id}`);
}

export function getMachineHistory(
  id: string,
  limit = 50
): Promise<SnapshotSummary[]> {
  return fetchJson(`${BASE}/machines/${id}/history?limit=${limit}`);
}

export function queueCommand(machineId: string, command: string): Promise<CommandRecord> {
  return fetch(`${BASE}/machines/${machineId}/commands`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  }).then(res => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });
}

export function getCommands(machineId: string): Promise<CommandRecord[]> {
  return fetchJson(`${BASE}/machines/${machineId}/commands`);
}

import type { MachineListItem, MachineDetail, SnapshotSummary } from "./types";

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

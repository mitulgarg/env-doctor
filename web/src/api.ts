import type {
  CommandActivityFilters,
  CommandActivityRow,
  CommandRecord,
  MachineDetail,
  MachineListItem,
  SnapshotSummary,
} from "./types";

const BASE = "/api";
const TOKEN_KEY = "envDoctorToken";

// When the server injects the token into the HTML, auto-store it so the login
// screen is skipped entirely. This runs before React mounts.
(function seedInjectedToken() {
  const injected = (window as unknown as Record<string, unknown>).__ENV_DOCTOR_TOKEN__;
  if (typeof injected === "string" && injected) {
    try { localStorage.setItem(TOKEN_KEY, injected); } catch { /* ignore */ }
  }
})();

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

class UnauthorizedError extends Error {
  constructor() {
    super("Unauthorized");
    this.name = "UnauthorizedError";
  }
}

async function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(url, { ...init, headers });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("envdoctor:unauthorized"));
    throw new UnauthorizedError();
  }
  return res;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await apiFetch(url);
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

export async function queueCommand(machineId: string, command: string): Promise<CommandRecord> {
  const res = await apiFetch(`${BASE}/machines/${machineId}/commands`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export function getCommands(machineId: string): Promise<CommandRecord[]> {
  return fetchJson(`${BASE}/machines/${machineId}/commands`);
}

export function getCommandActivity(
  filters: CommandActivityFilters = {}
): Promise<CommandActivityRow[]> {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.machine_id) params.set("machine_id", filters.machine_id);
  if (filters.since) params.set("since", filters.since);
  if (filters.limit != null) params.set("limit", String(filters.limit));
  if (filters.offset != null) params.set("offset", String(filters.offset));
  const qs = params.toString();
  return fetchJson(`${BASE}/commands${qs ? `?${qs}` : ""}`);
}

export async function verifyToken(): Promise<boolean> {
  try {
    const res = await apiFetch(`${BASE}/machines`);
    return res.ok;
  } catch {
    return false;
  }
}

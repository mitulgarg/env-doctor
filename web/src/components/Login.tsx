import { useState } from "react";
import { setToken } from "../api";

interface Props {
  onAuthenticated: () => void;
}

export default function Login({ onAuthenticated }: Props) {
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) {
      setError("Paste the API token printed by `env-doctor dashboard`.");
      return;
    }
    setBusy(true);
    setError(null);
    setToken(trimmed);
    // Probe a trivial endpoint to confirm the token works before unlocking.
    try {
      const res = await fetch("/api/machines", {
        headers: { Authorization: `Bearer ${trimmed}` },
      });
      if (res.status === 401) {
        setError("Token rejected. Check the value printed on the dashboard host.");
        setBusy(false);
        return;
      }
      if (!res.ok) {
        setError(`Server returned HTTP ${res.status}.`);
        setBusy(false);
        return;
      }
      onAuthenticated();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Network error";
      setError(`Could not reach the dashboard: ${msg}`);
      setBusy(false);
    }
  };

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      background: "#0d1117",
      color: "#e6edf3",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <form
        onSubmit={submit}
        style={{
          width: 380,
          padding: 28,
          background: "#161b22",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          boxShadow: "0 10px 40px rgba(0,0,0,0.4)",
        }}
      >
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
          env-doctor dashboard
        </div>
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 18 }}>
          Paste the API token printed by{" "}
          <code style={{ color: "rgba(255,255,255,0.7)" }}>env-doctor dashboard</code>{" "}
          on the host (also stored at <code>~/.env-doctor/api-token</code>).
        </div>

        <label htmlFor="token-input" style={{ fontSize: 11, color: "rgba(255,255,255,0.45)" }}>
          API token
        </label>
        <input
          id="token-input"
          type="password"
          value={value}
          onChange={e => setValue(e.target.value)}
          autoFocus
          spellCheck={false}
          autoComplete="off"
          style={{
            width: "100%",
            boxSizing: "border-box",
            marginTop: 6,
            padding: "10px 12px",
            background: "#0d1117",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 6,
            color: "#e6edf3",
            fontFamily: "monospace",
            fontSize: 13,
          }}
        />

        {error && (
          <div style={{ marginTop: 10, fontSize: 12, color: "#f47174" }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={busy}
          style={{
            marginTop: 18,
            width: "100%",
            padding: "10px 0",
            background: busy ? "#1f3a6e" : "#1f6feb",
            border: "none",
            color: "#fff",
            fontWeight: 600,
            fontSize: 13,
            borderRadius: 6,
            cursor: busy ? "default" : "pointer",
          }}
        >
          {busy ? "Verifying…" : "Unlock"}
        </button>
      </form>
    </div>
  );
}

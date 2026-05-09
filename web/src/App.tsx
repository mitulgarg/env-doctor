import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { clearToken, getMachines, getToken } from "./api";
import Login from "./components/Login";

function TopologyIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="9" cy="9" r="3" />
      <circle cx="3" cy="4" r="2" />
      <circle cx="15" cy="4" r="2" />
      <circle cx="4" cy="15" r="2" />
      <circle cx="14" cy="14" r="2" />
      <line x1="7" y1="7" x2="4.5" y2="5.5" />
      <line x1="11" y1="7" x2="13.5" y2="5.5" />
      <line x1="7.5" y1="11.5" x2="5.5" y2="13.5" />
      <line x1="11" y1="11" x2="12.5" y2="12.5" />
    </svg>
  );
}

function FleetIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="2" width="14" height="14" rx="2" />
      <line x1="2" y1="7" x2="16" y2="7" />
      <line x1="2" y1="11" x2="16" y2="11" />
      <line x1="7" y1="2" x2="7" y2="16" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="2 9 5 9 7 4 11 14 13 9 16 9" />
    </svg>
  );
}

const navStyle = (isActive: boolean): React.CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 12px",
  borderRadius: 8,
  textDecoration: "none",
  fontSize: 14,
  fontWeight: isActive ? 600 : 400,
  color: isActive ? "#fff" : "rgba(255,255,255,0.6)",
  background: isActive ? "rgba(255,255,255,0.1)" : "transparent",
});

export default function App() {
  const [counts, setCounts] = useState({ pass: 0, warning: 0, fail: 0 });
  const [authed, setAuthed] = useState<boolean>(() => Boolean(getToken()));

  useEffect(() => {
    const onUnauthorized = () => setAuthed(false);
    window.addEventListener("envdoctor:unauthorized", onUnauthorized);
    return () => window.removeEventListener("envdoctor:unauthorized", onUnauthorized);
  }, []);

  useEffect(() => {
    if (!authed) return;
    const load = () => {
      getMachines().then(machines => {
        setCounts({
          pass: machines.filter(m => m.latest_status === "pass").length,
          warning: machines.filter(m => m.latest_status === "warning").length,
          fail: machines.filter(m => m.latest_status === "fail").length,
        });
      }).catch(() => {});
    };
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, [authed]);

  const signOut = () => {
    clearToken();
    setAuthed(false);
  };

  if (!authed) {
    return <Login onAuthenticated={() => setAuthed(true)} />;
  }

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <nav style={{
        width: 220,
        background: "#1a1a2e",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
      }}>
        <div style={{ padding: "20px 16px 16px", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
              <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
              <path d="M8 15v5"/>
              <circle cx="8" cy="21" r="1"/>
              <circle cx="19" cy="12" r="2"/>
              <path d="M19 14v3a4 4 0 0 1-4 4h-3"/>
            </svg>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#fff" }}>env-doctor</div>
          </div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginTop: 4 }}>Fleet Dashboard</div>
        </div>

        <div style={{ padding: "12px 8px", display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
          <NavLink to="/topology" style={({ isActive }) => navStyle(isActive)}>
            <TopologyIcon /> Topology
          </NavLink>
          <NavLink to="/fleet" style={({ isActive }) => navStyle(isActive)}>
            <FleetIcon /> Fleet
          </NavLink>
          <NavLink to="/activity" style={({ isActive }) => navStyle(isActive)}>
            <ActivityIcon /> Activity
          </NavLink>
        </div>

        <div style={{
          padding: "12px 16px",
          borderTop: "1px solid rgba(255,255,255,0.08)",
          display: "flex",
          flexDirection: "column",
          gap: 10,
          alignItems: "center",
        }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "100%" }}>
            {[
              { color: "#40c057", count: counts.pass, label: "Healthy" },
              { color: "#fab005", count: counts.warning, label: "Warning" },
              { color: "#fa5252", count: counts.fail, label: "Failed" },
            ].map(({ color, count, label }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, boxShadow: `0 0 5px ${color}` }} />
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.45)" }}>{label}</span>
                </div>
                <span style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.7)" }}>{count}</span>
              </div>
            ))}
          </div>
          <a
            href="https://github.com/mitulgarg/env-doctor/issues"
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.25)",
              textDecoration: "none",
              letterSpacing: 0.2,
            }}
            onMouseEnter={e => (e.currentTarget.style.color = "rgba(255,255,255,0.55)")}
            onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.25)")}
          >
            Feedback / Issues ↗
          </a>
          <button
            type="button"
            onClick={signOut}
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.25)",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              letterSpacing: 0.2,
            }}
            onMouseEnter={e => (e.currentTarget.style.color = "rgba(255,255,255,0.55)")}
            onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.25)")}
          >
            Sign out
          </button>
        </div>
      </nav>

      <main style={{
        flex: 1,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        background: "#0d1117",
        color: "#e6edf3",
      }}>
        <Outlet />
      </main>
    </div>
  );
}

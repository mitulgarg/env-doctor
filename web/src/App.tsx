import { Link, Outlet } from "react-router-dom";

const styles = {
  app: {
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    maxWidth: 1200,
    margin: "0 auto",
    padding: "0 24px",
    color: "#1a1a2e",
    background: "#f8f9fa",
    minHeight: "100vh",
  } as React.CSSProperties,
  nav: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "16px 0",
    borderBottom: "1px solid #dee2e6",
    marginBottom: 24,
  } as React.CSSProperties,
  logo: {
    fontSize: 20,
    fontWeight: 700,
    color: "#1a1a2e",
    textDecoration: "none",
  } as React.CSSProperties,
  subtitle: {
    fontSize: 13,
    color: "#868e96",
  } as React.CSSProperties,
};

export default function App() {
  return (
    <div style={styles.app}>
      <nav style={styles.nav}>
        <Link to="/" style={styles.logo}>
          env-doctor
        </Link>
        <span style={styles.subtitle}>Fleet Dashboard</span>
      </nav>
      <Outlet />
    </div>
  );
}

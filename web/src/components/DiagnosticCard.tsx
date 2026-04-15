import type { CheckResult } from "../types";
import StatusBadge from "./StatusBadge";
import CopyCommand from "./CopyCommand";

const statusToLabel: Record<string, string> = {
  success: "pass",
  warning: "warning",
  error: "fail",
  not_found: "fail",
};

interface Props {
  title: string;
  result: CheckResult | null;
}

export default function DiagnosticCard({ title, result }: Props) {
  if (!result) return null;

  const status = statusToLabel[result.status] ?? result.status;

  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
        background: "rgba(255,255,255,0.04)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <h3 style={{ margin: 0, fontSize: 15, color: "#e6edf3" }}>{title}</h3>
        <StatusBadge status={status} />
      </div>

      {result.version && (
        <div style={{ fontSize: 13, color: "rgba(255,255,255,0.6)", marginBottom: 4 }}>
          Version: <strong style={{ color: "#e6edf3" }}>{result.version}</strong>
        </div>
      )}
      {result.path && (
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 4 }}>
          Path: {result.path}
        </div>
      )}

      {result.issues.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#f85149", marginBottom: 4 }}>
            Issues:
          </div>
          {result.issues.map((issue, i) => (
            <div key={i} style={{ fontSize: 13, color: "rgba(255,255,255,0.7)", paddingLeft: 8 }}>
              · {issue}
            </div>
          ))}
        </div>
      )}

      {result.recommendations.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#58a6ff", marginBottom: 4 }}>
            Recommendations:
          </div>
          {result.recommendations.map((rec, i) => {
            const isCommand =
              rec.startsWith("pip ") ||
              rec.startsWith("conda ") ||
              rec.startsWith("sudo ") ||
              rec.startsWith("apt ") ||
              rec.startsWith("curl ");
            return isCommand ? (
              <CopyCommand key={i} command={rec} />
            ) : (
              <div key={i} style={{ fontSize: 13, color: "rgba(255,255,255,0.6)", paddingLeft: 8 }}>
                {rec}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

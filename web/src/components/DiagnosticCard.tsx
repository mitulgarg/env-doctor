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
        border: "1px solid #dee2e6",
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
        background: "#fff",
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
        <h3 style={{ margin: 0, fontSize: 15 }}>{title}</h3>
        <StatusBadge status={status} />
      </div>

      {result.version && (
        <div style={{ fontSize: 13, color: "#495057", marginBottom: 4 }}>
          Version: <strong>{result.version}</strong>
        </div>
      )}
      {result.path && (
        <div style={{ fontSize: 12, color: "#868e96", marginBottom: 4 }}>
          Path: {result.path}
        </div>
      )}

      {result.issues.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#c92a2a", marginBottom: 4 }}>
            Issues:
          </div>
          {result.issues.map((issue, i) => (
            <div key={i} style={{ fontSize: 13, color: "#495057", paddingLeft: 8 }}>
              {issue}
            </div>
          ))}
        </div>
      )}

      {result.recommendations.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#1971c2", marginBottom: 4 }}>
            Recommendations:
          </div>
          {result.recommendations.map((rec, i) => {
            // If it looks like a command, render as copyable
            const isCommand =
              rec.startsWith("pip ") ||
              rec.startsWith("conda ") ||
              rec.startsWith("sudo ") ||
              rec.startsWith("apt ") ||
              rec.startsWith("curl ");
            return isCommand ? (
              <CopyCommand key={i} command={rec} />
            ) : (
              <div key={i} style={{ fontSize: 13, color: "#495057", paddingLeft: 8 }}>
                {rec}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

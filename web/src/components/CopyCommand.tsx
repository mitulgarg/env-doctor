import { useState } from "react";

export default function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback: do nothing
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "#161b22",
        color: "#e6edf3",
        padding: "6px 12px",
        borderRadius: 6,
        fontFamily: "monospace",
        fontSize: 13,
        cursor: "pointer",
        marginBottom: 4,
        border: "1px solid rgba(255,255,255,0.08)",
      }}
      onClick={handleCopy}
      title="Click to copy"
    >
      <code style={{ flex: 1 }}>{command}</code>
      <span style={{ fontSize: 11, color: copied ? "#51cf66" : "rgba(255,255,255,0.4)" }}>
        {copied ? "Copied!" : "Copy"}
      </span>
    </div>
  );
}

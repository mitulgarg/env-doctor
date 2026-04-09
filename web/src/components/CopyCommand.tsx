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
        background: "#212529",
        color: "#adb5bd",
        padding: "6px 12px",
        borderRadius: 6,
        fontFamily: "monospace",
        fontSize: 13,
        cursor: "pointer",
        marginBottom: 4,
      }}
      onClick={handleCopy}
      title="Click to copy"
    >
      <code style={{ flex: 1 }}>{command}</code>
      <span style={{ fontSize: 11, color: copied ? "#51cf66" : "#868e96" }}>
        {copied ? "Copied!" : "Copy"}
      </span>
    </div>
  );
}

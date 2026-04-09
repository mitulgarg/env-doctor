const colors: Record<string, { bg: string; fg: string }> = {
  pass: { bg: "#d3f9d8", fg: "#2b8a3e" },
  warning: { bg: "#fff3bf", fg: "#e67700" },
  fail: { bg: "#ffe3e3", fg: "#c92a2a" },
};

export default function StatusBadge({ status }: { status: string | null }) {
  const s = status?.toLowerCase() ?? "unknown";
  const c = colors[s] ?? { bg: "#e9ecef", fg: "#495057" };
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 10px",
        borderRadius: 12,
        fontSize: 12,
        fontWeight: 600,
        background: c.bg,
        color: c.fg,
        textTransform: "uppercase",
        letterSpacing: 0.5,
      }}
    >
      {s}
    </span>
  );
}

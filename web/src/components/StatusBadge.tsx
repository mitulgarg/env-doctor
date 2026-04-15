const colors: Record<string, { bg: string; fg: string }> = {
  pass: { bg: "#238636", fg: "#fff" },
  warning: { bg: "#d29922", fg: "#fff" },
  fail: { bg: "#da3633", fg: "#fff" },
};

export default function StatusBadge({ status }: { status: string | null }) {
  const s = status?.toLowerCase() ?? "unknown";
  const c = colors[s] ?? { bg: "rgba(255,255,255,0.12)", fg: "rgba(255,255,255,0.7)" };
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

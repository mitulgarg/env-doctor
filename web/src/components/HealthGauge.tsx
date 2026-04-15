interface Props {
  score: number; // 0-100
  size?: number;
}

export default function HealthGauge({ score, size = 130 }: Props) {
  const s = Math.max(0, Math.min(100, Math.round(score)));
  const color = s >= 80 ? "#238636" : s >= 40 ? "#d29922" : "#da3633";

  // SVG arc parameters — 270° arc, starting from bottom-left
  const cx = 60, cy = 60, r = 48;
  const startAngle = 135; // degrees
  const totalDegrees = 270;

  function polarToXY(angle: number): [number, number] {
    const rad = (angle * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  }

  function arcPath(startDeg: number, endDeg: number): string {
    const [sx, sy] = polarToXY(startDeg);
    const [ex, ey] = polarToXY(endDeg);
    const largeArc = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey}`;
  }

  const endAngle = startAngle + (s / 100) * totalDegrees;
  const trackPath = arcPath(startAngle, startAngle + totalDegrees);
  const fillPath = s > 0 ? arcPath(startAngle, endAngle) : "";

  const scoreLabel = s >= 80 ? "Healthy" : s >= 40 ? "Degraded" : "Critical";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg width={size} height={size} viewBox="0 0 120 120">
        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        {fillPath && (
          <path
            d={fillPath}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />
        )}
        {/* Score */}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="22"
          fontWeight="700"
          fill="#e6edf3"
        >
          {s}
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          fontSize="9"
          fill="rgba(255,255,255,0.4)"
        >
          / 100
        </text>
      </svg>
      <div style={{ fontSize: 12, fontWeight: 600, color, letterSpacing: 0.5 }}>
        {scoreLabel}
      </div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>Health Score</div>
    </div>
  );
}

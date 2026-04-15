import { useEffect, useRef } from "react";

interface Props {
  pass: number;
  warning: number;
  fail: number;
}

const COLORS = {
  pass: "#238636",
  warning: "#d29922",
  fail: "#da3633",
  empty: "rgba(255,255,255,0.08)",
};

export default function PieChart({ pass, warning, fail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = 160;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = devicePixelRatio || 1;
    const cx = (size * dpr) / 2;
    const cy = (size * dpr) / 2;
    const r = (size * dpr) / 2 - 10 * dpr;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const total = pass + warning + fail;

    if (total === 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.empty;
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.font = `${12 * dpr}px -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No data", cx, cy);
      return;
    }

    const segments: { value: number; color: string; label: string }[] = [
      { value: pass, color: COLORS.pass, label: "Pass" },
      { value: warning, color: COLORS.warning, label: "Warn" },
      { value: fail, color: COLORS.fail, label: "Fail" },
    ].filter(s => s.value > 0);

    let startAngle = -Math.PI / 2;
    for (const seg of segments) {
      const sweep = (seg.value / total) * Math.PI * 2;
      const endAngle = startAngle + sweep;

      // Arc
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, startAngle, endAngle);
      ctx.closePath();
      ctx.fillStyle = seg.color;
      ctx.fill();

      // Gap between segments
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r + 2 * dpr, startAngle, endAngle);
      ctx.strokeStyle = "#0d1117";
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();

      // Percentage label
      const midAngle = startAngle + sweep / 2;
      const labelR = r * 0.65;
      const lx = cx + Math.cos(midAngle) * labelR;
      const ly = cy + Math.sin(midAngle) * labelR;
      const pct = Math.round((seg.value / total) * 100);
      if (pct >= 8) {
        ctx.fillStyle = "#fff";
        ctx.font = `bold ${11 * dpr}px -apple-system, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${pct}%`, lx, ly);
      }

      startAngle = endAngle;
    }

    // Inner donut hole
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.45, 0, Math.PI * 2);
    ctx.fillStyle = "#0d1117";
    ctx.fill();

    // Center label
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = `bold ${14 * dpr}px -apple-system, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(total), cx, cy - 6 * dpr);
    ctx.font = `${9 * dpr}px -apple-system, sans-serif`;
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillText("machines", cx, cy + 9 * dpr);
  }, [pass, warning, fail]);

  const dpr = typeof devicePixelRatio !== "undefined" ? devicePixelRatio : 1;

  return (
    <canvas
      ref={canvasRef}
      width={size * dpr}
      height={size * dpr}
      style={{ width: size, height: size }}
    />
  );
}

import { useEffect, useRef, useState, useCallback } from "react";
import { getMachines, getMachine } from "../api";
import type { MachineListItem, MachineDetail, CheckResult } from "../types";

/* ─── constants ─── */
const BG = "#0d1117";
const GRID_DOT = "rgba(255,255,255,0.04)";
const TEXT_COL = "#e6edf3";
const STATUS_FILL: Record<string, string> = {
  pass: "#238636", warning: "#d29922", fail: "#da3633",
};
const STATUS_GLOW: Record<string, string> = {
  pass: "#40c057", warning: "#fab005", fail: "#fa5252",
};
const CENTER_FILL = "#0d1117";
const CENTER_GLOW = "#58a6ff";
const NODE_R = 22;
const CENTER_R = 32;
const IDEAL_DIST = 220;
const REFRESH_MS = 30_000;
const LERP = 0.07;

/* ─── types ─── */
interface GNode {
  id: string;
  x: number; y: number;
  vx: number; vy: number;
  hostname: string;
  status: string;
  platform: string | null;
  gpu: string | null;
  driver: string | null;
  cuda: string | null;
  torch: string | null;
  lastSeen: string | null;
  isCenter: boolean;
  r: number;
}

interface Camera {
  x: number; y: number; zoom: number;
  tx: number; ty: number; tz: number;
}

interface DragInfo {
  nodeId: string;
  ox: number; oy: number;
  sx: number; sy: number;
  moved: boolean;
}

/* ─── coord transforms ─── */
function s2w(sx: number, sy: number, c: Camera, w: number, h: number): [number, number] {
  return [(sx - w / 2) / c.zoom + c.x, (sy - h / 2) / c.zoom + c.y];
}
function w2s(wx: number, wy: number, c: Camera, w: number, h: number): [number, number] {
  return [(wx - c.x) * c.zoom + w / 2, (wy - c.y) * c.zoom + h / 2];
}

/* ─── GPU tier helpers ─── */
type GpuTier = "budget" | "mid" | "high" | "datacenter";

const GPU_DIMS: Record<GpuTier, { w: number; h: number }> = {
  budget:     { w: 54, h: 30 },
  mid:        { w: 64, h: 36 },
  high:       { w: 74, h: 42 },
  datacenter: { w: 90, h: 52 },
};

function classifyGpu(gpu: string | null): GpuTier {
  if (!gpu) return "budget";
  const g = gpu.toLowerCase();
  if (/a100|h100|v100|a800|h800|l40|a40|rtx\s*6000|rtx\s*8000|tesla/.test(g)) return "datacenter";
  if (/rtx\s*[34]090|rtx\s*[34]080|rtx\s*3090|titan/.test(g)) return "high";
  if (/rtx\s*[34]0[67]0|rtx\s*3060|rtx\s*2080|rtx\s*2070|gtx\s*1080/.test(g)) return "mid";
  return "budget";
}

function gpuShortName(gpu: string | null): string {
  if (!gpu) return "GPU";
  return gpu
    .replace(/nvidia\s+geforce\s+/i, "")
    .replace(/nvidia\s+/i, "")
    .replace(/geforce\s+/i, "")
    .trim()
    .slice(0, 14);
}

function roundRect(
  ctx: CanvasRenderingContext2D, x: number, y: number,
  w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/* ─── logo image (env-doctor stethoscope, inlined as data URL) ─── */
let _logoImg: HTMLImageElement | null = null;
const LOGO_DATA_URL =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none"' +
    ' stroke="#00ff88" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>' +
    '<path d="M8 15v5"/>' +
    '<circle cx="8" cy="21" r="1"/>' +
    '<circle cx="19" cy="12" r="2"/>' +
    '<path d="M19 14v3a4 4 0 0 1-4 4h-3"/>' +
    "</svg>",
  );

function getLogoImg(): HTMLImageElement {
  if (!_logoImg) {
    _logoImg = new Image();
    _logoImg.src = LOGO_DATA_URL;
  }
  return _logoImg;
}

/* ─── hit test ─── */
function hitTest(nodes: GNode[], wx: number, wy: number): GNode | null {
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (n.isCenter) {
      const dx = wx - n.x, dy = wy - n.y;
      if (dx * dx + dy * dy <= n.r * n.r * 1.8) return n;
    } else {
      const { w, h } = GPU_DIMS[classifyGpu(n.gpu)];
      if (Math.abs(wx - n.x) <= w * 0.6 && Math.abs(wy - n.y) <= h * 0.6) return n;
    }
  }
  return null;
}

/* ─── force simulation ─── */
function simulate(nodes: GNode[]) {
  const center = nodes.find(n => n.isCenter);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = b.x - a.x, dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; dist = 1; }
      const f = 3500 / (dist * dist);
      const fx = (dx / dist) * f, fy = (dy / dist) * f;
      if (!a.isCenter) { a.vx -= fx; a.vy -= fy; }
      if (!b.isCenter) { b.vx += fx; b.vy += fy; }
    }
  }

  if (center) {
    for (const n of nodes) {
      if (n.isCenter) continue;
      const dx = center.x - n.x, dy = center.y - n.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (dist - IDEAL_DIST) * 0.004;
      n.vx += (dx / dist) * f;
      n.vy += (dy / dist) * f;
    }
  }

  for (const n of nodes) {
    if (n.isCenter) continue; // hub position is set directly by drag; skip velocity
    n.vx *= 0.86;
    n.vy *= 0.86;
    n.x += n.vx;
    n.y += n.vy;
  }
}

/* ─── drawing helpers ─── */
function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number, cam: Camera) {
  const sp = 40;
  ctx.fillStyle = GRID_DOT;
  const [wx1, wy1] = s2w(0, 0, cam, w, h);
  const [wx2, wy2] = s2w(w, h, cam, w, h);
  const sx = Math.floor(wx1 / sp) * sp;
  const sy = Math.floor(wy1 / sp) * sp;
  for (let x = sx; x <= wx2; x += sp)
    for (let y = sy; y <= wy2; y += sp) {
      const [px, py] = w2s(x, y, cam, w, h);
      ctx.fillRect(px - 1, py - 1, 2, 2);
    }
}

function drawEdge(ctx: CanvasRenderingContext2D, a: GNode, b: GNode, cam: Camera, w: number, h: number, t: number) {
  const [ax, ay] = w2s(a.x, a.y, cam, w, h);
  const [bx, by] = w2s(b.x, b.y, cam, w, h);
  const pulse = 0.6 + 0.4 * Math.sin(t * 1.5 + b.x * 0.01);
  const grad = ctx.createLinearGradient(ax, ay, bx, by);
  grad.addColorStop(0, `rgba(88,166,255,${0.12 * pulse})`);
  const glow = STATUS_GLOW[b.status] ?? "#868e96";
  grad.addColorStop(1, glow + Math.round(40 * pulse).toString(16).padStart(2, "0"));
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.strokeStyle = grad;
  ctx.lineWidth = Math.max(1, 1.5 * cam.zoom);
  ctx.stroke();
}

function drawHubNode(
  ctx: CanvasRenderingContext2D, node: GNode, cam: Camera,
  w: number, h: number, hovered: boolean, sel: boolean, t: number,
) {
  const [sx, sy] = w2s(node.x, node.y, cam, w, h);
  const pulse = 0.85 + 0.15 * Math.sin(t * 2);
  const r = node.r * cam.zoom * (hovered ? 1.12 : 1);

  ctx.save();
  ctx.shadowColor = CENTER_GLOW;
  ctx.shadowBlur = (sel ? 32 : hovered ? 24 : 16) * pulse;
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.fillStyle = CENTER_FILL;
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.lineWidth = sel ? 3 : 2.5;
  ctx.strokeStyle = sel ? "#fff" : CENTER_GLOW;
  ctx.stroke();
  ctx.restore();

  // Logo image
  const logo = getLogoImg();
  const logoSize = r * 1.05;
  if (logo.complete && logo.naturalWidth > 0) {
    ctx.drawImage(logo, sx - logoSize / 2, sy - logoSize / 2, logoSize, logoSize);
  } else {
    // Fallback until loaded
    ctx.fillStyle = "#00ff88";
    ctx.font = `bold ${Math.round(r * 0.42)}px -apple-system, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("ED", sx, sy);
  }

}

function drawGpuNode(
  ctx: CanvasRenderingContext2D, node: GNode, cam: Camera,
  w: number, h: number, hovered: boolean, sel: boolean, t: number,
) {
  const [sx, sy] = w2s(node.x, node.y, cam, w, h);
  const pulse = 0.85 + 0.15 * Math.sin(t * 2 + node.x * 0.02);
  const tier = classifyGpu(node.gpu);
  const dims = GPU_DIMS[tier];
  const scale = cam.zoom * (hovered ? 1.08 : 1);
  const cw = dims.w * scale;
  const ch = dims.h * scale;
  const cx = sx - cw / 2;
  const cy = sy - ch / 2;
  const cr = Math.max(3, 5 * cam.zoom);

  const glowCol = STATUS_GLOW[node.status] ?? "#868e96";
  const statusFill = STATUS_FILL[node.status] ?? "#484f58";

  ctx.save();

  // Outer glow
  ctx.shadowColor = glowCol;
  ctx.shadowBlur = (sel ? 28 : hovered ? 20 : 12) * pulse;
  roundRect(ctx, cx, cy, cw, ch, cr);
  ctx.fillStyle = "#161b22";
  ctx.fill();
  ctx.shadowBlur = 0;

  // Border
  roundRect(ctx, cx, cy, cw, ch, cr);
  ctx.strokeStyle = sel ? "#ffffff" : glowCol + (hovered ? "cc" : "88");
  ctx.lineWidth = sel ? 2.5 : 1.5;
  ctx.stroke();

  ctx.restore();

  // Heatsink fins (consumer cards)
  if (tier !== "datacenter") {
    const finCount = tier === "budget" ? 5 : tier === "mid" ? 6 : 7;
    const finW = Math.max(1, 1.5 * cam.zoom);
    const finH = ch * 0.22;
    const finAreaW = cw - 10 * cam.zoom;
    const finGap = finAreaW / (finCount - 1);
    ctx.fillStyle = glowCol + "55";
    for (let i = 0; i < finCount; i++) {
      const fx = cx + 5 * cam.zoom + i * finGap - finW / 2;
      ctx.fillRect(fx, cy + 4 * cam.zoom, finW, finH);
    }
  }

  // Datacenter: horizontal slot lines
  if (tier === "datacenter") {
    ctx.strokeStyle = glowCol + "40";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      const ly = cy + (ch / 3) * i;
      ctx.beginPath();
      ctx.moveTo(cx + cr, ly);
      ctx.lineTo(cx + cw - cr, ly);
      ctx.stroke();
    }
    // LED dot
    ctx.fillStyle = glowCol;
    ctx.beginPath();
    ctx.arc(cx + cw - 8 * cam.zoom, cy + 8 * cam.zoom, 3 * cam.zoom, 0, Math.PI * 2);
    ctx.fill();
  }

  // Status strip at bottom
  const stripH = Math.max(3, 4 * cam.zoom);
  const stripY = cy + ch - stripH - 2 * cam.zoom;
  roundRect(ctx, cx + cr, stripY, cw - cr * 2, stripH, 2);
  ctx.fillStyle = statusFill;
  ctx.fill();

  // GPU model name
  const shortName = gpuShortName(node.gpu);
  const fontSize = Math.max(7, Math.round(9 * cam.zoom));
  ctx.fillStyle = "#e6edf3";
  ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Clip text to card interior so it never bleeds past the border
  ctx.save();
  roundRect(ctx, cx + 4, cy + 4, cw - 8, ch - 8, Math.max(1, cr - 2));
  ctx.clip();
  ctx.fillText(shortName, sx, sy + cam.zoom);
  ctx.restore();

  // Hostname below card
  ctx.font = `${Math.max(10, Math.round(12 * cam.zoom))}px -apple-system, sans-serif`;
  ctx.fillStyle = TEXT_COL;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(node.hostname, sx, cy + ch + 6);

  // GPU full name on deep zoom
  if (cam.zoom > 1.8 && node.gpu) {
    ctx.font = `${Math.round(9 * cam.zoom)}px -apple-system, sans-serif`;
    ctx.fillStyle = "rgba(230,237,243,0.45)";
    ctx.fillText(node.gpu, sx, cy + ch + 6 + Math.round(15 * cam.zoom));
  }
}

/* ─── detail panel helpers ─── */
function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function platformLabel(p: string | null): string {
  if (!p) return "Unknown";
  const l = p.toLowerCase();
  if (l.includes("windows")) return "Windows";
  if (l.includes("linux")) return "Linux";
  if (l.includes("darwin") || l.includes("mac")) return "macOS";
  return p;
}

/* ─── main component ─── */
export default function TopologyView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<GNode[]>([]);
  const camRef = useRef<Camera>({ x: 0, y: 0, zoom: 1, tx: 0, ty: 0, tz: 1 });
  const dragRef = useRef<DragInfo | null>(null);
  const panRef = useRef<{ sx: number; sy: number; cx: number; cy: number } | null>(null);
  const hoveredRef = useRef<string | null>(null);
  const sizeRef = useRef({ w: 800, h: 600 });
  const animRef = useRef(0);
  const selectedRef = useRef<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<MachineDetail | null>(null);
  const [hubModalOpen, setHubModalOpen] = useState(false);

  useEffect(() => { selectedRef.current = selected; }, [selected]);

  // sync machines → graph nodes
  useEffect(() => {
    const sync = (machines: MachineListItem[]) => {
      const prev = nodesRef.current;
      const map = new Map(prev.filter(n => !n.isCenter).map(n => [n.id, n]));
      let center = prev.find(n => n.isCenter);
      if (!center) {
        center = {
          id: "__hub__", x: 0, y: 0, vx: 0, vy: 0,
          hostname: "Dashboard", status: "pass",
          platform: null, gpu: null, driver: null, cuda: null, torch: null, lastSeen: null,
          isCenter: true, r: CENTER_R,
        };
      }
      const total = machines.length || 1;
      const next: GNode[] = [center];
      machines.forEach((m, i) => {
        const ex = map.get(m.id);
        if (ex) {
          ex.hostname = m.hostname;
          ex.status = m.latest_status ?? "unknown";
          ex.platform = m.platform;
          ex.gpu = m.gpu_name;
          ex.driver = m.driver_version;
          ex.cuda = m.cuda_version;
          ex.torch = m.torch_version;
          ex.lastSeen = m.last_seen;
          next.push(ex);
        } else {
          const angle = (2 * Math.PI * i) / total;
          next.push({
            id: m.id,
            x: Math.cos(angle) * IDEAL_DIST + (Math.random() - 0.5) * 30,
            y: Math.sin(angle) * IDEAL_DIST + (Math.random() - 0.5) * 30,
            vx: 0, vy: 0,
            hostname: m.hostname,
            status: m.latest_status ?? "unknown",
            platform: m.platform,
            gpu: m.gpu_name,
            driver: m.driver_version,
            cuda: m.cuda_version,
            torch: m.torch_version,
            lastSeen: m.last_seen,
            isCenter: false, r: NODE_R,
          });
        }
      });
      nodesRef.current = next;
    };
    const load = () => getMachines().then(sync).catch(console.error);
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  // fetch detail on select
  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    getMachine(selected).then(setDetail).catch(() => setDetail(null));
  }, [selected]);

  // canvas sizing
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const fit = () => {
      const w = el.clientWidth, h = el.clientHeight;
      sizeRef.current = { w, h };
      const c = canvasRef.current;
      if (!c) return;
      const dpr = devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = h * dpr;
      c.style.width = w + "px";
      c.style.height = h + "px";
    };
    const obs = new ResizeObserver(fit);
    obs.observe(el);
    fit();
    return () => obs.disconnect();
  }, []);

  // wheel (native listener for passive:false)
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const cam = camRef.current;
      cam.tz = Math.max(0.3, Math.min(5, cam.tz * (e.deltaY > 0 ? 0.92 : 1.08)));
    };
    c.addEventListener("wheel", handler, { passive: false });
    return () => c.removeEventListener("wheel", handler);
  }, []);

  // animation loop
  useEffect(() => {
    const loop = () => {
      const canvas = canvasRef.current;
      if (!canvas) { animRef.current = requestAnimationFrame(loop); return; }
      const ctx = canvas.getContext("2d");
      if (!ctx) { animRef.current = requestAnimationFrame(loop); return; }
      const { w, h } = sizeRef.current;
      const dpr = devicePixelRatio || 1;
      const nodes = nodesRef.current;
      const cam = camRef.current;
      const sel = selectedRef.current;
      const t = Date.now() * 0.001;

      simulate(nodes);
      cam.x += (cam.tx - cam.x) * LERP;
      cam.y += (cam.ty - cam.y) * LERP;
      cam.zoom += (cam.tz - cam.zoom) * LERP;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      drawGrid(ctx, w, h, cam);

      const hub = nodes.find(n => n.isCenter);
      if (hub) for (const n of nodes) { if (!n.isCenter) drawEdge(ctx, hub, n, cam, w, h, t); }
      for (const n of nodes) {
        if (n.isCenter) drawHubNode(ctx, n, cam, w, h, hoveredRef.current === n.id, sel === n.id, t);
        else drawGpuNode(ctx, n, cam, w, h, hoveredRef.current === n.id, sel === n.id, t);
      }

      if (nodes.length <= 1) {
        ctx.fillStyle = "rgba(255,255,255,0.3)";
        ctx.font = "14px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No machines reporting yet.", w / 2, h / 2 + 60);
        ctx.font = "12px -apple-system, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.2)";
        ctx.fillText("Run: env-doctor check --report-to http://this-server:8765", w / 2, h / 2 + 82);
      }

      ctx.restore();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  // mouse helpers
  const mpos = useCallback((e: React.MouseEvent): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const [sx, sy] = mpos(e);
    const { w, h } = sizeRef.current;
    const cam = camRef.current;
    const [wx, wy] = s2w(sx, sy, cam, w, h);
    const hit = hitTest(nodesRef.current, wx, wy);
    if (hit) {
      dragRef.current = { nodeId: hit.id, ox: hit.x - wx, oy: hit.y - wy, sx, sy, moved: false };
    } else {
      panRef.current = { sx, sy, cx: cam.tx, cy: cam.ty };
    }
  }, [mpos]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const [sx, sy] = mpos(e);
    const { w, h } = sizeRef.current;
    const cam = camRef.current;
    const canvas = canvasRef.current!;

    if (dragRef.current) {
      const d = dragRef.current;
      if (Math.abs(sx - d.sx) > 3 || Math.abs(sy - d.sy) > 3) d.moved = true;
      if (d.moved) {
        const [wx, wy] = s2w(sx, sy, cam, w, h);
        const node = nodesRef.current.find(n => n.id === d.nodeId);
        if (node) { node.x = wx + d.ox; node.y = wy + d.oy; node.vx = 0; node.vy = 0; }
      }
      canvas.style.cursor = "grabbing";
      return;
    }
    if (panRef.current) {
      const p = panRef.current;
      cam.tx = p.cx - (sx - p.sx) / cam.zoom;
      cam.ty = p.cy - (sy - p.sy) / cam.zoom;
      canvas.style.cursor = "grabbing";
      return;
    }
    const [wx, wy] = s2w(sx, sy, cam, w, h);
    const hit = hitTest(nodesRef.current, wx, wy);
    hoveredRef.current = hit ? hit.id : null;
    canvas.style.cursor = hit ? "pointer" : "default";
  }, [mpos]);

  const onMouseUp = useCallback(() => {
    if (dragRef.current && !dragRef.current.moved) {
      const nid = dragRef.current.nodeId;
      const node = nodesRef.current.find(n => n.id === nid);
      // Hub click opens info modal
      if (node && node.isCenter) {
        setHubModalOpen(true);
      } else if (node) {
        if (selectedRef.current === nid) {
          setSelected(null);
          camRef.current.tx = 0; camRef.current.ty = 0; camRef.current.tz = 1;
        } else {
          setSelected(nid);
          camRef.current.tx = node.x; camRef.current.ty = node.y; camRef.current.tz = 2.8;
        }
      }
    }
    dragRef.current = null;
    panRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = "default";
  }, []);

  const handleClose = useCallback(() => {
    setSelected(null);
    camRef.current.tx = 0; camRef.current.ty = 0; camRef.current.tz = 1;
  }, []);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Escape" && selectedRef.current) handleClose();
  }, [handleClose]);

  const report = detail?.latest_report;
  const checks = report?.checks;

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, position: "relative", overflow: "hidden", outline: "none" }}
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { hoveredRef.current = null; dragRef.current = null; panRef.current = null; }}
        style={{ display: "block" }}
      />

      {selected && detail && (
        <>
          {/* left panel: hardware */}
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: 280,
            background: "rgba(13,17,23,0.93)", backdropFilter: "blur(16px)",
            borderRight: "1px solid rgba(255,255,255,0.08)",
            padding: "20px 16px", overflowY: "auto", color: TEXT_COL,
            animation: "slideL .25s ease-out",
          }}>
            <button onClick={handleClose} style={backBtnStyle}>← Back</button>

            <h2 style={{ margin: "0 0 4px", fontSize: 18, fontWeight: 700 }}>{detail.hostname}</h2>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.45)", marginBottom: 20 }}>
              {platformLabel(detail.platform)} · Python {detail.python_version}
            </div>

            <SectionLabel text="Hardware" />
            {[
              { k: "GPU", v: detail.gpu_name },
              { k: "Driver", v: detail.driver_version },
              { k: "CUDA", v: detail.cuda_version },
              { k: "PyTorch", v: detail.torch_version },
            ].map(i => <InfoCard key={i.k} label={i.k} value={i.v ?? "—"} />)}

            <div style={{ marginTop: 16 }} />
            <SectionLabel text="Status" />
            <InfoCard label="Last Seen" value={timeAgo(detail.last_seen)} />
            <InfoCard label="Machine ID" value={detail.id} mono />
          </div>

          {/* right panel: diagnostics + json */}
          <div style={{
            position: "absolute", right: 0, top: 0, bottom: 0, width: 340,
            background: "rgba(13,17,23,0.93)", backdropFilter: "blur(16px)",
            borderLeft: "1px solid rgba(255,255,255,0.08)",
            padding: "20px 16px", overflowY: "auto", color: TEXT_COL,
            animation: "slideR .25s ease-out",
          }}>
            <div style={{ marginBottom: 16 }}>
              <span style={{
                padding: "4px 12px", borderRadius: 12, fontSize: 12, fontWeight: 600,
                textTransform: "uppercase",
                background: STATUS_FILL[detail.latest_status ?? ""] ?? "#484f58",
                color: "#fff",
              }}>
                {detail.latest_status ?? "unknown"}
              </span>
            </div>

            {report && (
              <>
                <SectionLabel text="Summary" />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 16 }}>
                  {[
                    { k: "Driver", v: report.summary.driver },
                    { k: "CUDA", v: report.summary.cuda },
                    { k: "cuDNN", v: report.summary.cudnn },
                    { k: "Issues", v: String(report.summary.issues_count) },
                  ].map(i => <InfoCard key={i.k} label={i.k} value={i.v || "—"} />)}
                </div>
              </>
            )}

            {checks && (
              <>
                <SectionLabel text="Diagnostics" />
                <DarkCard title="GPU / Driver" r={checks.driver} />
                <DarkCard title="CUDA Toolkit" r={checks.cuda} />
                {checks.cudnn && <DarkCard title="cuDNN" r={checks.cudnn} />}
                {checks.wsl2 && <DarkCard title="WSL2" r={checks.wsl2} />}
                <DarkCard title="Python Compat" r={checks.python_compat} />
                {checks.libraries && Object.entries(checks.libraries).map(([lib, res]) => (
                  <DarkCard key={lib} title={lib} r={res} />
                ))}
              </>
            )}

            {report && <JsonToggle data={report} />}
          </div>
        </>
      )}

      {hubModalOpen && (
        <div
          onClick={() => setHubModalOpen(false)}
          style={{
            position: "absolute", inset: 0,
            background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 100,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: "#161b22",
              border: "1px solid rgba(88,166,255,0.35)",
              borderRadius: 14,
              padding: "28px 32px",
              maxWidth: 440,
              width: "90%",
              color: "#e6edf3",
              boxShadow: "0 0 40px rgba(88,166,255,0.15)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                <path d="M8 15v5"/>
                <circle cx="8" cy="21" r="1"/>
                <circle cx="19" cy="12" r="2"/>
                <path d="M19 14v3a4 4 0 0 1-4 4h-3"/>
              </svg>
              <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>env-doctor Hub</h2>
              <button
                onClick={() => setHubModalOpen(false)}
                style={{ marginLeft: "auto", background: "none", border: "none", color: "rgba(255,255,255,0.4)", cursor: "pointer", fontSize: 20, lineHeight: 1 }}
              >
                ×
              </button>
            </div>

            <p style={{ fontSize: 14, lineHeight: 1.7, color: "rgba(255,255,255,0.65)", margin: "0 0 14px" }}>
              This is your <strong style={{ color: "#e6edf3" }}>centralised dashboard node</strong> — the machine currently running the env-doctor dashboard server.
            </p>
            <p style={{ fontSize: 14, lineHeight: 1.7, color: "rgba(255,255,255,0.65)", margin: "0 0 20px" }}>
              Remote GPU machines connect by running:
            </p>
            <pre style={{
              background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#79c0ff",
              margin: "0 0 20px", overflowX: "auto",
            }}>
              env-doctor check --report-to http://&lt;this-ip&gt;:8765
            </pre>
            <p style={{ fontSize: 14, lineHeight: 1.7, color: "rgba(255,255,255,0.65)", margin: "0 0 20px" }}>
              Each machine that reports in appears as a GPU node on this graph. Click any node to inspect its diagnostics and run remediation commands remotely.
            </p>

            <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 16 }}>
              <a
                href="https://github.com/mitulgarg/env-doctor/issues"
                target="_blank"
                rel="noreferrer"
                style={{
                  padding: "8px 18px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                  background: "rgba(88,166,255,0.08)", border: "1px solid rgba(88,166,255,0.3)",
                  color: "#58a6ff", textDecoration: "none",
                  display: "inline-flex", alignItems: "center", gap: 8,
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
                Feedback &amp; Feature Requests ↗
              </a>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slideL { from { transform: translateX(-100%); opacity: 0 } to { transform: translateX(0); opacity: 1 } }
        @keyframes slideR { from { transform: translateX(100%); opacity: 0 } to { transform: translateX(0); opacity: 1 } }
      `}</style>
    </div>
  );
}

/* ─── shared panel components ─── */
const backBtnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.08)", border: "none", color: "#e6edf3",
  padding: "6px 12px", borderRadius: 6, cursor: "pointer", fontSize: 13, marginBottom: 16,
};

function SectionLabel({ text }: { text: string }) {
  return (
    <div style={{
      fontSize: 11, color: "rgba(255,255,255,0.35)", textTransform: "uppercase",
      letterSpacing: 1, marginBottom: 8,
    }}>
      {text}
    </div>
  );
}

function InfoCard({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ padding: "10px 12px", background: "rgba(255,255,255,0.04)", borderRadius: 8, marginBottom: 6 }}>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>{label}</div>
      <div style={{
        fontSize: mono ? 11 : 14, fontWeight: mono ? 400 : 600, marginTop: 2,
        fontFamily: mono ? "monospace" : "inherit", wordBreak: "break-all",
      }}>
        {value}
      </div>
    </div>
  );
}

const STATUS_MAP: Record<string, string> = {
  success: "#238636", warning: "#d29922", error: "#da3633", not_found: "#484f58",
};

function DarkCard({ title, r }: { title: string; r: CheckResult }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: 12, marginBottom: 8,
      border: `1px solid ${(STATUS_MAP[r.status] ?? "rgba(255,255,255,0.08)") + "44"}`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{title}</span>
        <span style={{
          fontSize: 10, padding: "2px 8px", borderRadius: 10,
          background: STATUS_MAP[r.status] ?? "#484f58", color: "#fff",
          fontWeight: 600, textTransform: "uppercase",
        }}>
          {r.status}
        </span>
      </div>
      {r.version && <div style={{ fontSize: 12, color: "rgba(255,255,255,0.55)" }}>v{r.version}</div>}
      {r.issues.length > 0 && (
        <div style={{ marginTop: 6 }}>
          {r.issues.map((iss, i) => (
            <div key={i} style={{ fontSize: 11, color: "#f85149", marginTop: 2 }}>· {iss}</div>
          ))}
        </div>
      )}
      {r.recommendations.length > 0 && (
        <div style={{ marginTop: 4 }}>
          {r.recommendations.map((rec, i) => (
            <div key={i} style={{ fontSize: 11, color: "#58a6ff", marginTop: 2 }}>→ {rec}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function JsonToggle({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 16 }}>
      <button onClick={() => setOpen(!open)} style={{
        background: "rgba(255,255,255,0.08)", border: "none", color: "#e6edf3",
        padding: "8px 14px", borderRadius: 6, cursor: "pointer", fontSize: 12, width: "100%", textAlign: "left",
      }}>
        {open ? "▾" : "▸"} Raw JSON
      </button>
      {open && (
        <pre style={{
          background: "rgba(0,0,0,0.3)", padding: 12, borderRadius: 6,
          fontSize: 10, lineHeight: 1.5, overflow: "auto", maxHeight: 400,
          marginTop: 8, color: "rgba(255,255,255,0.65)",
        }}>
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import { getGroups, getMachine, getMachines, updateMachineGroup } from "../api";
import type { CheckResult, MachineDetail, MachineGroup, MachineListItem } from "../types";
import GroupPicker from "../components/GroupPicker";
import SelectionActionBar from "../components/SelectionActionBar";

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
// Same-group attractive force tuning. GROUP_IDEAL is the target distance between
// nodes that share a group; k_GROUP scales the spring stiffness. Values picked
// to nudge clustering without overpowering the global repulsion (3500/dist²).
const GROUP_IDEAL = 110;
const K_GROUP = 0.0008;
// Bubble centroid is lerped each frame so the background bubble doesn't jitter
// every time a node moves under the simulation.
const BUBBLE_LERP = 0.08;
// Synthetic name reserved for machines with group_name = null (mirrors backend).
const UNGROUPED_LABEL = "ungrouped";

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
  group: string | null;
  isCenter: boolean;
  r: number;
}

interface BubbleState {
  cx: number; cy: number; r: number;
}

interface LassoInfo {
  sx: number; sy: number; cx: number; cy: number;
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

/* ─── group colour ─── */
// Stable per-group hue from a string hash — same name → same colour every load.
function groupHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffffffff;
  return Math.abs(h) % 360;
}
function groupFill(name: string, alpha: number): string {
  return `hsla(${groupHue(name)}, 60%, 55%, ${alpha})`;
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

  // Same-group attraction: nodes sharing a group are gently pulled toward
  // GROUP_IDEAL distance from each other, producing visual clusters without
  // overpowering the global repulsion.
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    if (a.isCenter || !a.group) continue;
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      if (b.isCenter || a.group !== b.group) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (dist - GROUP_IDEAL) * K_GROUP;
      const fx = (dx / dist) * f, fy = (dy / dist) * f;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
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
  inMultiSelect: boolean = false, dim: boolean = false,
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

  // Outer dim wrapper so heatsink/text/labels also fade for non-search-matches.
  ctx.save();
  if (dim) ctx.globalAlpha = 0.35;

  ctx.save();

  // Multi-select highlight ring (drawn behind the card so it haloes it).
  if (inMultiSelect) {
    ctx.save();
    ctx.shadowColor = "#58a6ff";
    ctx.shadowBlur = 20 * pulse;
    roundRect(ctx, cx - 5, cy - 5, cw + 10, ch + 10, cr + 4);
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();
  }

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

  ctx.restore(); // closes the outer dim wrapper
}

function drawGroupBubble(
  ctx: CanvasRenderingContext2D, name: string, b: BubbleState,
  cam: Camera, w: number, h: number,
) {
  const [sx, sy] = w2s(b.cx, b.cy, cam, w, h);
  const r = b.r * cam.zoom;
  ctx.save();
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.fillStyle = groupFill(name, 0.08);
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = groupFill(name, 0.4);
  ctx.stroke();
  // Label above the bubble.
  ctx.font = `600 ${Math.max(11, Math.round(12 * cam.zoom))}px -apple-system, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = groupFill(name, 0.85);
  ctx.fillText(name, sx, sy - r - 6);
  ctx.restore();
}

function drawLasso(ctx: CanvasRenderingContext2D, l: LassoInfo) {
  const x = Math.min(l.sx, l.cx), y = Math.min(l.sy, l.cy);
  const w = Math.abs(l.cx - l.sx), h = Math.abs(l.cy - l.sy);
  ctx.save();
  ctx.fillStyle = "rgba(88,166,255,0.10)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "rgba(88,166,255,0.7)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  ctx.restore();
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

  // Phase 3 — group filter, search, multi-select, right-click context menu.
  const [groups, setGroups] = useState<MachineGroup[]>([]);
  const [groupFilter, setGroupFilter] = useState<string | null>(null); // null = "All", "ungrouped" = synthetic
  const [searchQuery, setSearchQuery] = useState("");
  const [creatingGroup, setCreatingGroup] = useState(false);
  const [newGroupName, setNewGroupName] = useState("");
  // Groups created in the UI before any node is assigned. They appear in the
  // dropdown so the user can immediately assign nodes to them; they "persist"
  // implicitly once the first PATCH lands and getGroups() picks them up.
  const [pendingGroups, setPendingGroups] = useState<string[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [contextMenu, setContextMenu] = useState<{ nodeId: string; sx: number; sy: number } | null>(null);
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [createdHint, setCreatedHint] = useState<string | null>(null);

  // Refs the animation loop reads (state in refs to avoid re-render churn).
  const groupFilterRef = useRef<string | null>(null);
  const searchQueryRef = useRef<string>("");
  const selectedIdsRef = useRef<Set<string>>(new Set());
  const lassoRef = useRef<LassoInfo | null>(null);
  const bubblesRef = useRef<Map<string, BubbleState>>(new Map());

  useEffect(() => { selectedRef.current = selected; }, [selected]);
  useEffect(() => { groupFilterRef.current = groupFilter; }, [groupFilter]);
  useEffect(() => { searchQueryRef.current = searchQuery.trim().toLowerCase(); }, [searchQuery]);
  useEffect(() => { selectedIdsRef.current = selectedIds; }, [selectedIds]);

  // Combined dropdown source: server-known groups + UI-created pending names.
  const dropdownGroups = useMemo<MachineGroup[]>(() => {
    const known = new Set(groups.map(g => g.name.toLowerCase()));
    const extras: MachineGroup[] = pendingGroups
      .filter(p => !known.has(p.toLowerCase()))
      .map(name => ({ name, machine_count: 0, status_breakdown: { pass: 0, warning: 0, fail: 0 } }));
    return [...groups.filter(g => g.name !== UNGROUPED_LABEL), ...extras];
  }, [groups, pendingGroups]);

  const refreshGroups = useCallback(() => {
    getGroups().then(setGroups).catch(() => {});
  }, []);
  useEffect(() => { refreshGroups(); }, [refreshGroups]);

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
          group: null, isCenter: true, r: CENTER_R,
        };
      }
      const total = machines.length || 1;
      const next: GNode[] = [center];
      machines.forEach((m, i) => {
        const ex = map.get(m.id);
        if (ex) {
          ex.hostname = m.hostname;
          ex.status = m.latest_status ?? "unknown";
          ex.group = m.group_name;
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
            group: m.group_name,
            isCenter: false, r: NODE_R,
          });
        }
      });
      nodesRef.current = next;
    };
    const load = () => {
      getMachines().then(sync).catch(console.error);
      getGroups().then(setGroups).catch(() => {});
    };
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
      const allNodes = nodesRef.current;
      const cam = camRef.current;
      const sel = selectedRef.current;
      const filter = groupFilterRef.current;
      const search = searchQueryRef.current;
      const selSet = selectedIdsRef.current;
      const t = Date.now() * 0.001;

      // Visibility: hub always shown; others honour the group filter.
      // null filter = "All", "ungrouped" = nodes with group === null.
      const isVisible = (n: GNode): boolean => {
        if (n.isCenter) return true;
        if (filter == null) return true;
        if (filter === UNGROUPED_LABEL) return n.group == null;
        return n.group === filter;
      };
      const visibleNodes = allNodes.filter(isVisible);

      // Simulate only visible — hidden nodes don't tug from off-screen.
      simulate(visibleNodes);
      cam.x += (cam.tx - cam.x) * LERP;
      cam.y += (cam.ty - cam.y) * LERP;
      cam.zoom += (cam.tz - cam.zoom) * LERP;

      // Group bubble centroid + radius (lerped to dampen jitter).
      // Skip bubble drawing entirely when filter narrows to a single group.
      const showBubbles = filter == null;
      const bubbles = bubblesRef.current;
      if (showBubbles) {
        const byGroup = new Map<string, GNode[]>();
        for (const n of visibleNodes) {
          if (n.isCenter || !n.group) continue;
          const list = byGroup.get(n.group) ?? [];
          list.push(n);
          byGroup.set(n.group, list);
        }
        // Drop bubbles for groups that no longer exist.
        for (const k of bubbles.keys()) if (!byGroup.has(k)) bubbles.delete(k);
        for (const [name, members] of byGroup) {
          const cx = members.reduce((s, n) => s + n.x, 0) / members.length;
          const cy = members.reduce((s, n) => s + n.y, 0) / members.length;
          let r = 0;
          for (const n of members) {
            const d = Math.hypot(n.x - cx, n.y - cy);
            if (d > r) r = d;
          }
          r = Math.max(60, r + 36); // padding so the bubble surrounds the cards
          const prev = bubbles.get(name);
          if (!prev) {
            bubbles.set(name, { cx, cy, r });
          } else {
            prev.cx += (cx - prev.cx) * BUBBLE_LERP;
            prev.cy += (cy - prev.cy) * BUBBLE_LERP;
            prev.r += (r - prev.r) * BUBBLE_LERP;
          }
        }
      } else {
        bubbles.clear();
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      drawGrid(ctx, w, h, cam);

      // Bubbles below edges/nodes.
      if (showBubbles) {
        for (const [name, b] of bubbles) drawGroupBubble(ctx, name, b, cam, w, h);
      }

      const hub = visibleNodes.find(n => n.isCenter);
      if (hub) for (const n of visibleNodes) { if (!n.isCenter) drawEdge(ctx, hub, n, cam, w, h, t); }
      for (const n of visibleNodes) {
        const matchesSearch = !search || n.hostname.toLowerCase().includes(search);
        const dim = !!search && !matchesSearch;
        if (n.isCenter) {
          drawHubNode(ctx, n, cam, w, h, hoveredRef.current === n.id, sel === n.id, t);
        } else {
          drawGpuNode(
            ctx, n, cam, w, h,
            hoveredRef.current === n.id, sel === n.id, t,
            selSet.has(n.id), dim,
          );
        }
      }

      // Lasso overlay (in screen coords — drawn last, no transform).
      if (lassoRef.current) drawLasso(ctx, lassoRef.current);

      if (allNodes.length <= 1) {
        ctx.fillStyle = "rgba(255,255,255,0.3)";
        ctx.font = "14px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No machines reporting yet.", w / 2, h / 2 + 60);
        ctx.font = "12px -apple-system, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,0.2)";
        const host = window.location.hostname;
        ctx.fillText(`Run: env-doctor check --report-to http://${host}:8765`, w / 2, h / 2 + 82);
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
    if (e.button === 2) return; // right-click handled by onContextMenu
    e.preventDefault();
    const [sx, sy] = mpos(e);
    const { w, h } = sizeRef.current;
    const cam = camRef.current;
    const [wx, wy] = s2w(sx, sy, cam, w, h);
    const hit = hitTest(nodesRef.current, wx, wy);

    // Close any open right-click menu when the user starts a new gesture.
    setContextMenu(null);

    // Shift modifier: multi-select toggle (on hit) or lasso (on empty canvas).
    if (e.shiftKey) {
      if (hit && !hit.isCenter) {
        setSelectedIds(prev => {
          const next = new Set(prev);
          if (next.has(hit.id)) next.delete(hit.id); else next.add(hit.id);
          return next;
        });
      } else if (!hit) {
        lassoRef.current = { sx, sy, cx: sx, cy: sy };
      }
      return;
    }

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

    if (lassoRef.current) {
      lassoRef.current.cx = sx;
      lassoRef.current.cy = sy;
      canvas.style.cursor = "crosshair";
      return;
    }
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
    canvas.style.cursor = e.shiftKey ? "crosshair" : (hit ? "pointer" : "default");
  }, [mpos]);

  const onMouseUp = useCallback(() => {
    // Resolve lasso: bulk-select every visible non-center node whose screen
    // position lies inside the rectangle.
    if (lassoRef.current) {
      const l = lassoRef.current;
      const minX = Math.min(l.sx, l.cx), maxX = Math.max(l.sx, l.cx);
      const minY = Math.min(l.sy, l.cy), maxY = Math.max(l.sy, l.cy);
      // Only treat as a lasso if the user actually dragged (not a misfired shift+click).
      if (maxX - minX > 4 || maxY - minY > 4) {
        const { w, h } = sizeRef.current;
        const cam = camRef.current;
        const filter = groupFilterRef.current;
        const hits: string[] = [];
        for (const n of nodesRef.current) {
          if (n.isCenter) continue;
          // Honour the group filter — invisible nodes can't be lassoed.
          if (filter != null) {
            if (filter === UNGROUPED_LABEL && n.group != null) continue;
            if (filter !== UNGROUPED_LABEL && n.group !== filter) continue;
          }
          const [px, py] = w2s(n.x, n.y, cam, w, h);
          if (px >= minX && px <= maxX && py >= minY && py <= maxY) hits.push(n.id);
        }
        if (hits.length > 0) {
          setSelectedIds(prev => {
            const next = new Set(prev);
            for (const id of hits) next.add(id);
            return next;
          });
        }
      }
      lassoRef.current = null;
    }

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

  const onContextMenu = useCallback((e: React.MouseEvent) => {
    const [sx, sy] = mpos(e);
    const { w, h } = sizeRef.current;
    const cam = camRef.current;
    const [wx, wy] = s2w(sx, sy, cam, w, h);
    const hit = hitTest(nodesRef.current, wx, wy);
    if (hit && !hit.isCenter) {
      e.preventDefault();
      setContextMenu({ nodeId: hit.id, sx, sy });
    }
  }, [mpos]);

  const handleClose = useCallback(() => {
    setSelected(null);
    camRef.current.tx = 0; camRef.current.ty = 0; camRef.current.tz = 1;
  }, []);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key !== "Escape") return;
    if (contextMenu) { setContextMenu(null); return; }
    if (selectedIds.size > 0) { setSelectedIds(new Set()); return; }
    if (selectedRef.current) handleClose();
  }, [handleClose, contextMenu, selectedIds]);

  // Bulk-assign helper used by the SelectionActionBar and the right-click menu.
  const assignGroupBulk = useCallback(async (ids: string[], group: string | null) => {
    if (ids.length === 0) return;
    setBulkError(null);
    try {
      await Promise.all(ids.map(id => updateMachineGroup(id, group)));
      // Optimistically update the in-memory nodes so clustering reacts on the
      // next animation frame without waiting for the next 30s machine poll.
      const set = new Set(ids);
      for (const n of nodesRef.current) {
        if (!n.isCenter && set.has(n.id)) n.group = group;
      }
      // Also sync the open detail panel if its machine was in the batch.
      if (detail && set.has(detail.id)) {
        setDetail({ ...detail, group_name: group });
      }
      // If we just used a pending group, it's now real — drop it from pending.
      if (group) setPendingGroups(prev => prev.filter(p => p !== group));
      refreshGroups();
    } catch (e: unknown) {
      setBulkError(e instanceof Error ? e.message : "Bulk update failed");
    }
  }, [detail, refreshGroups]);

  const handleSelectionAssign = useCallback(async (group: string | null) => {
    const ids = Array.from(selectedIds);
    await assignGroupBulk(ids, group);
    setSelectedIds(new Set());
  }, [assignGroupBulk, selectedIds]);

  const handleContextMenuAssign = useCallback(async (group: string | null) => {
    if (!contextMenu) return;
    await assignGroupBulk([contextMenu.nodeId], group);
    setContextMenu(null);
  }, [assignGroupBulk, contextMenu]);

  const handleCreateNewGroup = useCallback(() => {
    const name = newGroupName.trim();
    if (!name) { setCreatingGroup(false); return; }
    if (name.toLowerCase() === UNGROUPED_LABEL) {
      setBulkError("'ungrouped' is reserved");
      return;
    }
    setPendingGroups(prev => prev.includes(name) ? prev : [...prev, name]);
    // Reset filter to "All" so the user can see and select nodes to assign,
    // and surface a transient hint pointing them at the next step.
    setGroupFilter(null);
    setNewGroupName("");
    setCreatingGroup(false);
    setCreatedHint(name);
  }, [newGroupName]);

  // Auto-dismiss the "group created" hint after a few seconds.
  useEffect(() => {
    if (!createdHint) return;
    const id = setTimeout(() => setCreatedHint(null), 6000);
    return () => clearTimeout(id);
  }, [createdHint]);

  const report = detail?.latest_report;
  const checks = report?.checks;

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, position: "relative", overflow: "hidden", outline: "none", display: "flex", flexDirection: "column" }}
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      {/* Toolbar — Group filter + new group + search */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 16px",
        background: "rgba(13,17,23,0.85)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        flexShrink: 0,
        zIndex: 10,
        position: "relative",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>Group:</span>
          <select
            value={groupFilter ?? ""}
            onChange={e => setGroupFilter(e.target.value || null)}
            style={{
              padding: "6px 10px",
              background: "#0d1117",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 6,
              color: "#e6edf3",
              fontSize: 13,
              minWidth: 160,
            }}
          >
            <option value="">All ({groups.reduce((s, g) => s + g.machine_count, 0)})</option>
            {dropdownGroups.map(g => (
              <option key={g.name} value={g.name}>{g.name} ({g.machine_count})</option>
            ))}
            {groups.find(g => g.name === UNGROUPED_LABEL) && (
              <option value={UNGROUPED_LABEL}>
                Ungrouped ({groups.find(g => g.name === UNGROUPED_LABEL)?.machine_count ?? 0})
              </option>
            )}
          </select>
        </div>

        {creatingGroup ? (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="text"
              autoFocus
              value={newGroupName}
              onChange={e => setNewGroupName(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") handleCreateNewGroup();
                else if (e.key === "Escape") { setCreatingGroup(false); setNewGroupName(""); }
              }}
              placeholder="Group name…"
              style={{
                padding: "6px 10px",
                background: "#0d1117",
                border: "1px solid rgba(88,166,255,0.5)",
                borderRadius: 6,
                color: "#e6edf3",
                fontSize: 13,
                width: 180,
              }}
            />
            <button
              type="button"
              onClick={handleCreateNewGroup}
              style={{
                padding: "6px 12px",
                background: "#1f6feb",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >Add</button>
            <button
              type="button"
              onClick={() => { setCreatingGroup(false); setNewGroupName(""); }}
              style={{
                padding: "6px 8px",
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 6,
                color: "rgba(255,255,255,0.6)",
                fontSize: 12,
                cursor: "pointer",
              }}
            >✕</button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setCreatingGroup(true)}
            style={{
              padding: "6px 12px",
              background: "transparent",
              border: "1px dashed rgba(255,255,255,0.25)",
              borderRadius: 6,
              color: "rgba(255,255,255,0.7)",
              fontSize: 12,
              cursor: "pointer",
            }}
            title="Create a new group (assign nodes via shift+click + Assign group)"
          >+ New group</button>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>🔍</span>
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search hostname…"
            style={{
              padding: "6px 10px",
              background: "#0d1117",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 6,
              color: "#e6edf3",
              fontSize: 13,
              width: 200,
            }}
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              style={{
                padding: "6px 8px",
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 6,
                color: "rgba(255,255,255,0.6)",
                fontSize: 12,
                cursor: "pointer",
              }}
            >✕</button>
          )}
        </div>

        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginLeft: 8 }} title="Hold Shift to multi-select or lasso, right-click a node for quick group assignment">
          Shift+click / drag · Right-click
        </div>
      </div>

      <div style={{ flex: 1, position: "relative" }}>
      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onContextMenu={onContextMenu}
        onMouseLeave={() => { hoveredRef.current = null; dragRef.current = null; panRef.current = null; }}
        style={{ display: "block" }}
      />

      {createdHint && (
        <div style={{
          position: "absolute",
          top: 16,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(31,111,235,0.95)",
          color: "#fff",
          padding: "10px 18px",
          borderRadius: 8,
          fontSize: 13,
          fontWeight: 500,
          boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
          zIndex: 35,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}>
          <span>
            Group <strong style={{ fontWeight: 700 }}>"{createdHint}"</strong> created.
            Shift-click or lasso nodes to add them, or right-click a single node.
          </span>
          <button
            type="button"
            onClick={() => setCreatedHint(null)}
            style={{
              background: "transparent",
              border: "none",
              color: "rgba(255,255,255,0.85)",
              fontSize: 16,
              cursor: "pointer",
              lineHeight: 1,
              padding: 0,
            }}
            title="Dismiss"
          >×</button>
        </div>
      )}

      {selectedIds.size > 0 && (
        <SelectionActionBar
          count={selectedIds.size}
          groups={dropdownGroups}
          onAssign={handleSelectionAssign}
          onClear={() => setSelectedIds(new Set())}
          error={bulkError}
        />
      )}

      {contextMenu && (
        <div
          style={{
            position: "absolute",
            left: contextMenu.sx,
            top: contextMenu.sy,
            zIndex: 40,
            minWidth: 240,
          }}
        >
          <GroupPicker
            value={nodesRef.current.find(n => n.id === contextMenu.nodeId)?.group ?? null}
            groups={dropdownGroups}
            onChange={handleContextMenuAssign}
            onClose={() => setContextMenu(null)}
          />
        </div>
      )}

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
            <button onClick={handleClose} style={backBtnStyle}>✕ Close</button>

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
            <div style={{
              marginBottom: 16,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
            }}>
              <span style={{
                padding: "4px 12px", borderRadius: 12, fontSize: 12, fontWeight: 600,
                textTransform: "uppercase",
                background: STATUS_FILL[detail.latest_status ?? ""] ?? "#484f58",
                color: "#fff",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
              }}>
                {detail.latest_status ?? "unknown"}
                {report && (
                  <span style={{
                    fontSize: 11,
                    fontWeight: 500,
                    opacity: 0.85,
                    textTransform: "none",
                    letterSpacing: 0,
                  }}>
                    · {report.summary.issues_count} {report.summary.issues_count === 1 ? "issue" : "issues"}
                  </span>
                )}
              </span>
              <Link
                to={`/machines/${detail.id}`}
                style={{
                  padding: "6px 12px",
                  background: "#1f6feb",
                  color: "#fff",
                  borderRadius: 6,
                  fontSize: 12,
                  fontWeight: 600,
                  textDecoration: "none",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  transition: "background .15s",
                }}
                onMouseEnter={e => (e.currentTarget.style.background = "#388bfd")}
                onMouseLeave={e => (e.currentTarget.style.background = "#1f6feb")}
                title="Open the full details page (history, command runner, group editor)"
              >
                Open full details →
              </Link>
            </div>

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

      {/* Zoom controls */}
      <div style={{
        position: "absolute", bottom: 20, right: 20,
        display: "flex", flexDirection: "column", gap: 4,
        zIndex: 10,
      }}>
        {[
          { label: "+", delta: 1.25, title: "Zoom in" },
          { label: "−", delta: 0.8, title: "Zoom out" },
          { label: "⌂", delta: null, title: "Reset view" },
        ].map(({ label, delta, title }) => (
          <button
            key={label}
            title={title}
            onClick={() => {
              const cam = camRef.current;
              if (delta === null) {
                cam.tx = 0; cam.ty = 0; cam.tz = 1;
              } else {
                cam.tz = Math.max(0.3, Math.min(5, cam.tz * delta));
              }
            }}
            style={{
              width: 32, height: 32,
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.14)",
              borderRadius: 6,
              color: "#e6edf3",
              fontSize: label === "⌂" ? 16 : 18,
              cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              lineHeight: 1,
            }}
          >
            {label}
          </button>
        ))}
      </div>

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
      </div>

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

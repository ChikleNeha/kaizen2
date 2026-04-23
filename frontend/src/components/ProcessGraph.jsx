import { useRef, useEffect, useCallback } from 'react';

/**
 * ProcessGraph
 * Canvas-based animated node graph showing process relationships.
 *
 * Rendering rules (from spec Section 5.3)
 * ----------------------------------------
 * - Pure HTML5 Canvas — no D3, no react-flow, no graph libraries
 * - Nodes: filled circles with radial gradient glow
 * - Node colour: green=healthy, amber=warning, red=critical/chaos, indigo=protected
 * - Bad process node: pulsing outer ring + secondary dashed ring at r+4
 * - Node labels: process name centred, PID below in smaller text
 * - Edges: quadratic bezier curves with linear gradient stroke (60% opacity)
 * - Bad process edges: dashed [4,4] and red
 * - Travelling dot on each edge animating along path
 * - Kill animation: node + edges fade out over 40 frames
 * - All nodes connect to central CPU_core and RAM_bus hub nodes
 * - Background transparent — panel provides the surface
 */

// ---------------------------------------------------------------------------
// Colour constants — spec design system
// ---------------------------------------------------------------------------
const C = {
  green:  '#4ade80',
  amber:  '#f59e0b',
  red:    '#ef4444',
  indigo: '#818cf8',
  bg:     '#111111',
  card:   '#161616',
  border: '#1f1f1f',
  text:   '#e2e2e2',
  muted:  '#888888',
  dim:    '#444444',
};

// ---------------------------------------------------------------------------
// Layout — fixed node positions as ratios of canvas dimensions
// ---------------------------------------------------------------------------
function buildLayout(W, H) {
  return {
    CPU_core: { x: W * 0.50, y: H * 0.42, r: 18, label: 'CPU_core', sub: 'hub', isHub: true },
    RAM_bus:  { x: W * 0.50, y: H * 0.72, r: 14, label: 'RAM_bus',  sub: 'hub', isHub: true },
    p0: { x: W * 0.16, y: H * 0.20 },
    p1: { x: W * 0.38, y: H * 0.12 },
    p2: { x: W * 0.62, y: H * 0.12 },
    p3: { x: W * 0.84, y: H * 0.20 },
    p4: { x: W * 0.16, y: H * 0.65 },
    p5: { x: W * 0.84, y: H * 0.65 },
    p6: { x: W * 0.30, y: H * 0.88 },
    p7: { x: W * 0.70, y: H * 0.88 },
  };
}

const PROC_SLOTS = ['p0','p1','p2','p3','p4','p5','p6','p7'];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

function nodeColour(proc) {
  if (!proc) return C.dim;
  if (proc.is_protected) return C.indigo;
  if (proc.cpu_percent > 60) return C.red;
  if (proc.cpu_percent > 30) return C.amber;
  return C.green;
}

function isBadProc(proc) {
  return proc && !proc.is_protected && proc.cpu_percent > 60;
}

// Quadratic bezier point at parameter t
function bezierPoint(x0,y0, cx,cy, x1,y1, t) {
  const mt = 1 - t;
  return {
    x: mt*mt*x0 + 2*mt*t*cx + t*t*x1,
    y: mt*mt*y0 + 2*mt*t*cy + t*t*y1,
  };
}

// Control point for a quadratic bezier — arcs gently away from midpoint
function controlPoint(x0,y0,x1,y1, bend=0.25) {
  const mx = (x0+x1)/2, my = (y0+y1)/2;
  const dx = x1-x0, dy = y1-y0;
  return { cx: mx - dy*bend, cy: my + dx*bend };
}

// ---------------------------------------------------------------------------
// ProcessGraph component
// ---------------------------------------------------------------------------
export default function ProcessGraph({ processList = [], width = 700, height = 220 }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const stateRef  = useRef({
    t: 0,
    nodes: [],           // { slot, proc, colour, alpha, ring }
    killAlphas: {},      // pid → alpha (1→0 fade on kill)
    prevPids: new Set(),
  });

  // Build node list from processList prop
  useEffect(() => {
    const s = stateRef.current;
    const currentPids = new Set(processList.map(p => p.pid));

    // Detect killed PIDs — start fade-out
    for (const pid of s.prevPids) {
      if (!currentPids.has(pid) && s.killAlphas[pid] === undefined) {
        s.killAlphas[pid] = 1.0;
      }
    }

    // Build node list — up to 8 processes mapped to slots
    s.nodes = processList.slice(0, 8).map((proc, i) => ({
      slot: PROC_SLOTS[i],
      proc,
      colour: nodeColour(proc),
      alpha: 1.0,
    }));

    s.prevPids = currentPids;
  }, [processList]);

  // Animation loop
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const layout = buildLayout(W, H);
    const s = stateRef.current;

    s.t += 0.012;

    ctx.clearRect(0, 0, W, H);

    // ---- Hub nodes -------------------------------------------------------
    const hubs = [
      { key: 'CPU_core', ...layout.CPU_core },
      { key: 'RAM_bus',  ...layout.RAM_bus  },
    ];

    // ---- Process nodes ---------------------------------------------------
    const procNodes = s.nodes.map(n => ({
      ...n,
      ...layout[n.slot],
      r: 11,
    }));

    // Advance kill fade alphas
    for (const pid of Object.keys(s.killAlphas)) {
      s.killAlphas[pid] = Math.max(0, s.killAlphas[pid] - 0.025);
      if (s.killAlphas[pid] === 0) delete s.killAlphas[pid];
    }

    // ---- Draw edges from each process node to CPU_core and RAM_bus -------
    const cpu = layout.CPU_core;
    const ram = layout.RAM_bus;

    for (const n of procNodes) {
      if (!n.proc) continue;
      const killAlpha = s.killAlphas[n.proc.pid] ?? 1.0;
      const edgeAlpha = n.alpha * killAlpha;
      const bad = isBadProc(n.proc);

      // Edge to CPU_core
      _drawEdge(ctx, n.x, n.y, cpu.x, cpu.y, n.colour, C.green, bad, edgeAlpha, s.t);
      // Edge to RAM_bus
      _drawEdge(ctx, n.x, n.y, ram.x, ram.y, n.colour, C.amber, bad, edgeAlpha * 0.6, s.t + 1.2);
    }

    // ---- Draw hub nodes --------------------------------------------------
    for (const hub of hubs) {
      _drawHubNode(ctx, hub.x, hub.y, hub.r, hub.label, hub.key === 'CPU_core' ? C.green : C.amber, s.t);
    }

    // ---- Draw process nodes + fading kills ------------------------------
    for (const n of procNodes) {
      if (!n.proc) continue;
      const killAlpha = s.killAlphas[n.proc.pid] ?? 1.0;
      const finalAlpha = n.alpha * killAlpha;
      const bad = isBadProc(n.proc);
      _drawProcNode(ctx, n.x, n.y, n.r, n.proc, n.colour, finalAlpha, bad, s.t);
    }

    // ---- Draw ghost nodes for fading killed processes --------------------
    // (The process has been removed from s.nodes but killAlpha still > 0)
    // Nothing extra needed — handled by edgeAlpha + nodeAlpha above.

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ display: 'block', width: '100%', height: `${height}px` }}
    />
  );
}

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

function _drawEdge(ctx, x0, y0, x1, y1, colA, colB, bad, alpha, t) {
  if (alpha <= 0.01) return;

  const { cx, cy } = controlPoint(x0, y0, x1, y1, bad ? 0.35 : 0.18);

  // Gradient along the edge
  const grad = ctx.createLinearGradient(x0, y0, x1, y1);
  grad.addColorStop(0, hexAlpha(colA, 0.6 * alpha));
  grad.addColorStop(1, hexAlpha(colB, 0.6 * alpha));

  ctx.save();
  ctx.strokeStyle = grad;
  ctx.lineWidth = bad ? 1.2 : 0.8;

  if (bad) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = hexAlpha(C.red, 0.7 * alpha);
  } else {
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.quadraticCurveTo(cx, cy, x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Travelling dot along the edge
  const dotT = (t * 0.4 + (x0 * 0.001)) % 1.0;
  const dotPos = bezierPoint(x0, y0, cx, cy, x1, y1, dotT);
  ctx.save();
  ctx.globalAlpha = alpha * 0.9;
  ctx.fillStyle = bad ? C.red : colA;
  ctx.beginPath();
  ctx.arc(dotPos.x, dotPos.y, 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function _drawHubNode(ctx, x, y, r, label, colour, t) {
  ctx.save();

  // Outer ambient glow
  const glow = ctx.createRadialGradient(x, y, r * 0.3, x, y, r * 3.5);
  glow.addColorStop(0, hexAlpha(colour, 0.12));
  glow.addColorStop(1, hexAlpha(colour, 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, r * 3.5, 0, Math.PI * 2);
  ctx.fill();

  // Node body
  const bodyGrad = ctx.createRadialGradient(x - r*0.3, y - r*0.3, 0, x, y, r);
  bodyGrad.addColorStop(0, hexAlpha(colour, 0.3));
  bodyGrad.addColorStop(1, hexAlpha(colour, 0.08));
  ctx.fillStyle = bodyGrad;
  ctx.strokeStyle = hexAlpha(colour, 0.6);
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Pulsing ring
  const pulse = 0.5 + 0.5 * Math.sin(t * 2.5);
  ctx.strokeStyle = hexAlpha(colour, 0.3 * pulse);
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.arc(x, y, r + 4 + pulse * 3, 0, Math.PI * 2);
  ctx.stroke();

  // Label
  ctx.fillStyle = colour;
  ctx.font = `bold 8px ui-monospace, 'JetBrains Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x, y);

  ctx.restore();
}

function _drawProcNode(ctx, x, y, r, proc, colour, alpha, bad, t) {
  if (alpha <= 0.01) return;
  ctx.save();
  ctx.globalAlpha = alpha;

  // Outer glow (larger for bad processes)
  const glowR = bad ? r * 4 : r * 2.5;
  const glow = ctx.createRadialGradient(x, y, r * 0.5, x, y, glowR);
  glow.addColorStop(0, hexAlpha(colour, bad ? 0.20 : 0.08));
  glow.addColorStop(1, hexAlpha(colour, 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, Math.PI * 2);
  ctx.fill();

  // Node body
  const bodyGrad = ctx.createRadialGradient(x - r*0.3, y - r*0.3, 0, x, y, r);
  bodyGrad.addColorStop(0, hexAlpha(colour, 0.25));
  bodyGrad.addColorStop(1, C.card);
  ctx.fillStyle = bodyGrad;
  ctx.strokeStyle = hexAlpha(colour, bad ? 0.9 : 0.5);
  ctx.lineWidth = bad ? 1.0 : 0.5;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Bad process: pulsing outer ring + dashed ring
  if (bad) {
    const pulse = 0.5 + 0.5 * Math.sin(t * 3.5);
    // Pulsing ring
    ctx.strokeStyle = hexAlpha(C.red, 0.7 * pulse);
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(x, y, r + 3 + pulse * 2, 0, Math.PI * 2);
    ctx.stroke();

    // Secondary dashed ring at r+4
    ctx.strokeStyle = hexAlpha(C.red, 0.35);
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(x, y, r + 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Process name — centred in node
  const name = proc.name.length > 10
    ? proc.name.slice(0, 9) + '…'
    : proc.name;
  ctx.fillStyle = colour;
  ctx.font = `600 7px ui-monospace, 'JetBrains Mono', monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(name, x, y - 2);

  // PID below name
  ctx.fillStyle = hexAlpha(colour, 0.6);
  ctx.font = `500 6px ui-monospace, 'JetBrains Mono', monospace`;
  ctx.fillText(`${proc.pid}`, x, y + 6);

  // CPU% label below bad node only
  if (bad) {
    ctx.fillStyle = C.red;
    ctx.font = `bold 7px ui-monospace, 'JetBrains Mono', monospace`;
    ctx.fillText(`${proc.cpu_percent.toFixed(0)}%`, x, y + r + 10);
  }

  ctx.restore();
}

import { useRef, useEffect, useCallback } from 'react';

/**
 * ProcessGraph — v3 (final dissolve fix)
 *
 * Root cause of dissolve not working in v2:
 * processList is memoized in App.jsx. By the time the memo fires and
 * s.nodes rebuilds (pid removed), the ghost-seeding code in the
 * processList useEffect has nothing to snapshot — the node is already gone.
 *
 * Fix: seed ghost from lastAction prop instead.
 * lastAction (state.action) arrives in the same WS message as the new
 * processList. By putting its useEffect FIRST and keying it only on
 * lastAction, we snapshot the dying node from s.current.nodes while it
 * still contains the killed pid — before the processList memo updates.
 *
 * Props
 * -----
 * processList : array   — state.obs.process_list from WS
 * lastAction  : object  — state.action from WS  ← NEW, required for dissolve
 * width/height: number
 */

const C = {
  green:  '#4ade80',
  amber:  '#f59e0b',
  red:    '#ef4444',
  indigo: '#818cf8',
  card:   '#161616',
  dim:    '#444444',
};

function buildLayout(W, H) {
  return {
    CPU_core: { x: W*0.50, y: H*0.42, r: 18, label: 'CPU_core', isHub: true },
    RAM_bus:  { x: W*0.50, y: H*0.72, r: 14, label: 'RAM_bus',  isHub: true },
    p0: { x: W*0.16, y: H*0.20 },
    p1: { x: W*0.38, y: H*0.12 },
    p2: { x: W*0.62, y: H*0.12 },
    p3: { x: W*0.84, y: H*0.20 },
    p4: { x: W*0.16, y: H*0.65 },
    p5: { x: W*0.84, y: H*0.65 },
    p6: { x: W*0.30, y: H*0.88 },
    p7: { x: W*0.70, y: H*0.88 },
  };
}

const PROC_SLOTS = ['p0','p1','p2','p3','p4','p5','p6','p7'];

function hexAlpha(hex, a) {
  const rv = parseInt(hex.slice(1,3),16);
  const g  = parseInt(hex.slice(3,5),16);
  const b  = parseInt(hex.slice(5,7),16);
  return `rgba(${rv},${g},${b},${Math.max(0,Math.min(1,a))})`;
}

function nodeColour(proc) {
  if (!proc) return C.dim;
  if (proc.is_protected)      return C.indigo;
  if (proc.cpu_percent > 60)  return C.red;
  if (proc.cpu_percent > 30)  return C.amber;
  return C.green;
}

function isBadProc(proc) {
  return proc && !proc.is_protected && proc.cpu_percent > 60;
}

function bezierPoint(x0,y0,cx,cy,x1,y1,t) {
  const mt = 1-t;
  return { x: mt*mt*x0+2*mt*t*cx+t*t*x1, y: mt*mt*y0+2*mt*t*cy+t*t*y1 };
}

function controlPoint(x0,y0,x1,y1,bend=0.25) {
  const mx=(x0+x1)/2, my=(y0+y1)/2, dx=x1-x0, dy=y1-y0;
  return { cx: mx-dy*bend, cy: my+dx*bend };
}

// ---------------------------------------------------------------------------
export default function ProcessGraph({
  processList = [],
  lastAction  = null,
  width       = 700,
  height      = 220,
}) {
  const canvasRef   = useRef(null);
  const animRef     = useRef(null);
  const lastActRef  = useRef(null);  // prevents re-seeding same action twice

  // All mutable canvas state — never in React state (avoids re-render loops)
  const s = useRef({
    t:          0,
    nodes:      [],  // active nodes { slot, proc, colour, r, x, y }
    ghostNodes: {},  // pid → { x, y, r, proc, colour, alpha }
  });

  // ── EFFECT 1: Seed ghost from lastAction ─────────────────────────────────
  // Must run BEFORE Effect 2 (React runs effects top-to-bottom).
  // s.current.nodes still contains the killed node at this point.
  useEffect(() => {
    if (!lastAction || lastAction.tool_name !== 'kill_process') return;
    if (lastActRef.current === lastAction) return;  // same object ref = already processed
    lastActRef.current = lastAction;

    const killedPid = Number(lastAction.pid);
    if (!killedPid) return;

    const dying = s.current.nodes.find(n => n.proc && n.proc.pid === killedPid);
    if (!dying) {
      // Fallback: if node not found (e.g. first render), place ghost at a
      // default slot so something dissolves rather than nothing
      return;
    }

    // Don't overwrite a ghost that's mid-fade
    if (s.current.ghostNodes[killedPid] !== undefined) return;

    s.current.ghostNodes[killedPid] = {
      x:      dying.x,
      y:      dying.y,
      r:      dying.r ?? 11,
      proc:   { ...dying.proc },   // clone — processList may mutate original
      colour: dying.colour,
      alpha:  1.0,
    };
  }, [lastAction]);

  // ── EFFECT 2: Rebuild active nodes from processList ───────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width;
    const H = canvas.height;
    const layout = buildLayout(W, H);

    s.current.nodes = processList.slice(0, 8).map((proc, i) => {
      const slot = PROC_SLOTS[i];
      const pos  = layout[slot] ?? {};
      return { slot, proc, colour: nodeColour(proc), alpha: 1.0, r: 11, x: pos.x, y: pos.y };
    });
  }, [processList]);

  // ── Animation loop ────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx    = canvas.getContext('2d');
    const W      = canvas.width;
    const H      = canvas.height;
    const layout = buildLayout(W, H);
    const cur    = s.current;

    cur.t += 0.012;
    ctx.clearRect(0, 0, W, H);

    const cpu = layout.CPU_core;
    const ram = layout.RAM_bus;

    // Fade ghost nodes: 0.018/frame → ~55 frames → ~900ms dissolve at 60fps
    for (const pid of Object.keys(cur.ghostNodes)) {
      cur.ghostNodes[pid].alpha -= 0.018;
      if (cur.ghostNodes[pid].alpha <= 0) delete cur.ghostNodes[pid];
    }

    // ---- Edges: active nodes -------------------------------------------
    for (const n of cur.nodes) {
      if (!n.x) continue;
      const bad = isBadProc(n.proc);
      _drawEdge(ctx, n.x, n.y, cpu.x, cpu.y, n.colour, C.green, bad, 1.0,       cur.t);
      _drawEdge(ctx, n.x, n.y, ram.x, ram.y, n.colour, C.amber, bad, 0.6,       cur.t+1.2);
    }

    // ---- Edges: ghost nodes (fading red dashes) ------------------------
    for (const g of Object.values(cur.ghostNodes)) {
      _drawEdge(ctx, g.x, g.y, cpu.x, cpu.y, C.red, C.red, true, g.alpha*0.8,  cur.t);
      _drawEdge(ctx, g.x, g.y, ram.x, ram.y, C.red, C.red, true, g.alpha*0.4,  cur.t+1.2);
    }

    // ---- Hubs ----------------------------------------------------------
    _drawHubNode(ctx, cpu.x, cpu.y, cpu.r, 'CPU_core', C.green, cur.t);
    _drawHubNode(ctx, ram.x, ram.y, ram.r, 'RAM_bus',  C.amber, cur.t);

    // ---- Active process nodes ------------------------------------------
    for (const n of cur.nodes) {
      if (!n.x) continue;
      _drawProcNode(ctx, n.x, n.y, n.r, n.proc, n.colour, 1.0, isBadProc(n.proc), cur.t);
    }

    // ---- Ghost (dissolving) nodes — always rendered as bad=true --------
    for (const g of Object.values(cur.ghostNodes)) {
      _drawProcNode(ctx, g.x, g.y, g.r, g.proc, g.colour, g.alpha, true, cur.t);
    }

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
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
  ctx.save();
  ctx.lineWidth = bad ? 1.2 : 0.8;
  if (bad) {
    ctx.setLineDash([4,4]);
    ctx.strokeStyle = hexAlpha(C.red, 0.7*alpha);
  } else {
    ctx.setLineDash([]);
    const grad = ctx.createLinearGradient(x0,y0,x1,y1);
    grad.addColorStop(0, hexAlpha(colA, 0.6*alpha));
    grad.addColorStop(1, hexAlpha(colB, 0.6*alpha));
    ctx.strokeStyle = grad;
  }
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.quadraticCurveTo(cx,cy,x1,y1); ctx.stroke();
  ctx.setLineDash([]); ctx.restore();

  const dotT   = (t*0.4 + x0*0.001) % 1.0;
  const dotPos = bezierPoint(x0,y0,cx,cy,x1,y1,dotT);
  ctx.save();
  ctx.globalAlpha = alpha*0.9;
  ctx.fillStyle   = bad ? C.red : colA;
  ctx.beginPath(); ctx.arc(dotPos.x, dotPos.y, 1.5, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}

function _drawHubNode(ctx, x, y, r, label, colour, t) {
  ctx.save();
  const glow = ctx.createRadialGradient(x,y,r*0.3,x,y,r*3.5);
  glow.addColorStop(0, hexAlpha(colour,0.12)); glow.addColorStop(1, hexAlpha(colour,0));
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x,y,r*3.5,0,Math.PI*2); ctx.fill();

  const body = ctx.createRadialGradient(x-r*0.3,y-r*0.3,0,x,y,r);
  body.addColorStop(0, hexAlpha(colour,0.3)); body.addColorStop(1, hexAlpha(colour,0.08));
  ctx.fillStyle=body; ctx.strokeStyle=hexAlpha(colour,0.6); ctx.lineWidth=0.5;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.stroke();

  const pulse = 0.5+0.5*Math.sin(t*2.5);
  ctx.strokeStyle=hexAlpha(colour,0.3*pulse); ctx.lineWidth=0.5;
  ctx.beginPath(); ctx.arc(x,y,r+4+pulse*3,0,Math.PI*2); ctx.stroke();

  ctx.fillStyle=colour; ctx.font=`bold 8px ui-monospace,'JetBrains Mono',monospace`;
  ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(label,x,y);
  ctx.restore();
}

function _drawProcNode(ctx, x, y, r, proc, colour, alpha, bad, t) {
  if (alpha<=0.01) return;
  ctx.save(); ctx.globalAlpha=alpha;

  const glowR=bad?r*4:r*2.5;
  const glow=ctx.createRadialGradient(x,y,r*0.5,x,y,glowR);
  glow.addColorStop(0,hexAlpha(colour,bad?0.20:0.08)); glow.addColorStop(1,hexAlpha(colour,0));
  ctx.fillStyle=glow; ctx.beginPath(); ctx.arc(x,y,glowR,0,Math.PI*2); ctx.fill();

  const body=ctx.createRadialGradient(x-r*0.3,y-r*0.3,0,x,y,r);
  body.addColorStop(0,hexAlpha(colour,0.25)); body.addColorStop(1,C.card);
  ctx.fillStyle=body; ctx.strokeStyle=hexAlpha(colour,bad?0.9:0.5); ctx.lineWidth=bad?1.0:0.5;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.stroke();

  if (bad) {
    const pulse=0.5+0.5*Math.sin(t*3.5);
    ctx.strokeStyle=hexAlpha(C.red,0.7*pulse); ctx.lineWidth=0.8;
    ctx.beginPath(); ctx.arc(x,y,r+3+pulse*2,0,Math.PI*2); ctx.stroke();
    ctx.strokeStyle=hexAlpha(C.red,0.35); ctx.lineWidth=0.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.arc(x,y,r+7,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }

  const name=proc.name.length>10?proc.name.slice(0,9)+'…':proc.name;
  ctx.fillStyle=colour; ctx.font=`600 7px ui-monospace,'JetBrains Mono',monospace`;
  ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(name,x,y-2);
  ctx.fillStyle=hexAlpha(colour,0.6); ctx.font=`500 6px ui-monospace,'JetBrains Mono',monospace`;
  ctx.fillText(`${proc.pid}`,x,y+6);
  if (bad) {
    ctx.fillStyle=C.red; ctx.font=`bold 7px ui-monospace,'JetBrains Mono',monospace`;
    ctx.fillText(`${proc.cpu_percent.toFixed(0)}%`,x,y+r+10);
  }
  ctx.restore();
}
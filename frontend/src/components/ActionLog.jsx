import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * ActionLog
 * Right column — scrollable history of every action taken this episode.
 *
 * v2 changes
 * ----------
 * - Panel is horizontally resizable via a drag handle on the left edge
 * - Clicking a log entry expands it to show the full message (no truncation)
 * - Expanded entry shows full action JSON + full result message
 */

// ---------------------------------------------------------------------------
// Action type → colour mapping
// ---------------------------------------------------------------------------
const ACTION_COLOURS = {
  kill_process:       '#ef4444',
  thermal_mitigation: '#f59e0b',
  allocate_memory:    '#f59e0b',
  prioritize_task:    '#4ade80',
  inspect_logs:       '#888888',
  list_processes:     '#888888',
  wait:               '#444444',
  parse_error:        '#ef4444',
};

function actionColour(toolName) {
  return ACTION_COLOURS[toolName] ?? '#888888';
}

function actionIcon(toolName) {
  const icons = {
    kill_process:       '✕',
    thermal_mitigation: '▼',
    allocate_memory:    '◈',
    prioritize_task:    '↑',
    inspect_logs:       '⌕',
    list_processes:     '≡',
    wait:               '·',
    parse_error:        '!',
  };
  return icons[toolName] ?? '·';
}

function shortDesc(action) {
  if (!action?.tool_name) return '—';
  switch (action.tool_name) {
    case 'kill_process':       return `kill pid ${action.pid}`;
    case 'thermal_mitigation': return `thermal → ${action.strategy}`;
    case 'allocate_memory':    return `free ${action.mb_to_free}MB pid ${action.target_pid}`;
    case 'prioritize_task':    return `priority ${action.priority} pid ${action.pid}`;
    case 'inspect_logs':       return action.pid != null ? `logs pid ${action.pid}` : 'system logs';
    case 'list_processes':     return 'list processes';
    case 'wait':               return 'wait';
    case 'parse_error':        return 'parse error';
    default:                   return action.tool_name;
  }
}

function fullDesc(action) {
  if (!action?.tool_name) return '—';
  const lines = [`tool: ${action.tool_name}`];
  const skip = new Set(['tool_name']);
  for (const [k, v] of Object.entries(action)) {
    if (skip.has(k)) continue;
    const val = typeof v === 'string' && v.length > 60
      ? v.slice(0, 60) + '…'
      : String(v);
    lines.push(`${k}: ${val}`);
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Single log entry
// ---------------------------------------------------------------------------
function LogEntry({ entry, isNew }) {
  const [expanded, setExpanded] = useState(false);
  const colour  = actionColour(entry.action?.tool_name);
  const icon    = actionIcon(entry.action?.tool_name);
  const desc    = shortDesc(entry.action);
  const success = entry.result?.success;

  return (
    <div
      onClick={() => setExpanded(e => !e)}
      style={{
        padding:         '6px 10px',
        borderBottom:    '0.5px solid #1f1f1f',
        animation:       isNew ? 'kaizen-fadein 0.3s ease forwards' : 'none',
        backgroundColor: entry.action?.tool_name === 'kill_process' && success
          ? 'rgba(239,68,68,0.04)'
          : 'transparent',
        cursor:          'pointer',
        transition:      'background-color 0.15s ease',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.backgroundColor =
          entry.action?.tool_name === 'kill_process' && success
            ? 'rgba(239,68,68,0.08)'
            : '#1a1a1a';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.backgroundColor =
          entry.action?.tool_name === 'kill_process' && success
            ? 'rgba(239,68,68,0.04)'
            : 'transparent';
      }}
    >
      {/* ── Step + tool row ───────────────────────────────────── */}
      <div style={{
        display:    'flex',
        alignItems: 'center',
        gap:        '6px',
        marginBottom: expanded ? '6px' : '2px',
      }}>
        {/* Step badge */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:   '8px',
          color:      '#444444',
          minWidth:   '20px',
          flexShrink: 0,
        }}>
          s{entry.step}
        </span>

        {/* Icon */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:   '10px',
          color:      colour,
          width:      '10px',
          textAlign:  'center',
          flexShrink: 0,
        }}>
          {icon}
        </span>

        {/* Action description — truncated when collapsed */}
        <span style={{
          fontFamily:   "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:     '10px',
          color:        colour,
          whiteSpace:   expanded ? 'normal' : 'nowrap',
          overflow:     expanded ? 'visible' : 'hidden',
          textOverflow: expanded ? 'clip'    : 'ellipsis',
          flex:         1,
          wordBreak:    expanded ? 'break-all' : 'normal',
        }}>
          {desc}
        </span>

        {/* Reward delta */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:   '9px',
          color:      entry.reward > 0 ? '#4ade80'
                    : entry.reward < 0 ? '#ef4444'
                    : '#444444',
          flexShrink: 0,
          minWidth:   '36px',
          textAlign:  'right',
        }}>
          {entry.reward > 0 ? '+' : ''}{entry.reward.toFixed(2)}
        </span>

        {/* Expand chevron */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:   '8px',
          color:      '#2a2a2a',
          flexShrink: 0,
          marginLeft: '2px',
          transition: 'transform 0.2s ease',
          transform:  expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          display:    'inline-block',
        }}>
          ▾
        </span>
      </div>

      {/* ── Collapsed: short result message ───────────────────── */}
      {!expanded && entry.result?.message && (
        <div style={{
          fontFamily:   "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:     '8px',
          color:        success === false ? '#ef4444' : '#444444',
          paddingLeft:  '36px',
          whiteSpace:   'nowrap',
          overflow:     'hidden',
          textOverflow: 'ellipsis',
          opacity:      0.8,
        }}>
          {entry.result.message.split('\n')[0]}
        </div>
      )}

      {/* ── Expanded: full action details + full message ──────── */}
      {expanded && (
        <div style={{
          marginTop:   '4px',
          paddingLeft: '36px',
          animation:   'kaizen-fadein 0.2s ease forwards',
        }}>
          {/* Full action params */}
          <pre style={{
            fontFamily:      "ui-monospace, 'JetBrains Mono', monospace",
            fontSize:        '8px',
            color:           colour,
            opacity:         0.85,
            whiteSpace:      'pre-wrap',
            wordBreak:       'break-all',
            backgroundColor: 'rgba(255,255,255,0.02)',
            border:          '0.5px solid #1f1f1f',
            borderRadius:    '3px',
            padding:         '5px 6px',
            marginBottom:    '4px',
            lineHeight:      1.6,
          }}>
            {fullDesc(entry.action)}
          </pre>

          {/* Full result message */}
          {entry.result?.message && (
            <pre style={{
              fontFamily:      "ui-monospace, 'JetBrains Mono', monospace",
              fontSize:        '8px',
              color:           success === false ? '#ef4444' : '#888888',
              whiteSpace:      'pre-wrap',
              wordBreak:       'break-all',
              backgroundColor: 'rgba(255,255,255,0.02)',
              border:          '0.5px solid #1f1f1f',
              borderRadius:    '3px',
              padding:         '5px 6px',
              lineHeight:      1.6,
            }}>
              {entry.result.message}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Resize handle
// ---------------------------------------------------------------------------
function ResizeHandle({ onDrag }) {
  const dragging = useRef(false);

  const onMouseDown = useCallback((e) => {
    e.preventDefault();
    dragging.current = true;

    const onMouseMove = (ev) => {
      if (!dragging.current) return;
      onDrag(ev.clientX);
    };
    const onMouseUp = () => {
      dragging.current = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup',   onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup',   onMouseUp);
  }, [onDrag]);

  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        position:   'absolute',
        left:       0,
        top:        0,
        bottom:     0,
        width:      '6px',
        cursor:     'col-resize',
        zIndex:     10,
        display:    'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Visual grip dots */}
      <div style={{
        width:        '2px',
        height:       '24px',
        borderRadius: '1px',
        background:   'repeating-linear-gradient(to bottom, #2a2a2a 0px, #2a2a2a 2px, transparent 2px, transparent 4px)',
      }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
const MIN_WIDTH = 120;
const MAX_WIDTH = 360;
const DEFAULT_WIDTH = 180;

export default function ActionLog({ state }) {
  const [entries,    setEntries]    = useState([]);
  const [newStepId,  setNewStepId]  = useState(null);
  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH);

  const lastEpisodeRef = useRef(null);
  const lastStepRef    = useRef(null);
  const scrollRef      = useRef(null);
  const panelRef       = useRef(null);

  // ---- Resize handler --------------------------------------------------
  const handleDrag = useCallback((clientX) => {
    if (!panelRef.current) return;
    const rect  = panelRef.current.getBoundingClientRect();
    const newW  = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, rect.right - clientX));
    setPanelWidth(newW);
  }, []);

  // ---- Entry accumulation ----------------------------------------------
  useEffect(() => {
    if (!state) return;

    const episode = state.episode;
    const step    = state.step;

    if (episode !== lastEpisodeRef.current) {
      setEntries([]);
      lastEpisodeRef.current = episode;
      lastStepRef.current    = null;
    }

    if (step === lastStepRef.current) return;
    lastStepRef.current = step;

    const entryId = `${episode}-${step}`;
    setNewStepId(entryId);

    setEntries(prev => {
      if (prev.some(e => e.id === entryId)) return prev;
      return [
        ...prev,
        {
          id:      entryId,
          step,
          episode,
          action:  state.action,
          result:  state.action_result,
          reward:  state.reward ?? 0,
        },
      ];
    });

    const t = setTimeout(() => setNewStepId(null), 400);
    return () => clearTimeout(t);
  }, [state]);

  // ---- Auto-scroll to bottom ------------------------------------------
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const cumulative = entries.reduce((sum, e) => sum + e.reward, 0);

  return (
    <div
      ref={panelRef}
      style={{
        width:         `${panelWidth}px`,
        flexShrink:    0,
        position:      'relative',
        display:       'flex',
        flexDirection: 'column',
        overflow:      'hidden',
        transition:    'none',
      }}
    >
      {/* Resize handle on left edge */}
      <ResizeHandle onDrag={handleDrag} />

      {/* Left border (sits after handle so it's visible) */}
      <div style={{
        position: 'absolute',
        left:     6,
        top:      0,
        bottom:   0,
        width:    '0.5px',
        backgroundColor: '#1f1f1f',
        pointerEvents: 'none',
      }} />

      {/* ── Header ──────────────────────────────────────────────── */}
      <div style={{
        padding:       '7px 10px 7px 16px',
        borderBottom:  '0.5px solid #1f1f1f',
        display:       'flex',
        justifyContent: 'space-between',
        alignItems:    'center',
        flexShrink:    0,
      }}>
        <span style={{
          fontFamily:    "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:      '9px',
          color:         '#444444',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Action Log
        </span>
        {entries.length > 0 && (
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize:   '9px',
            color:      cumulative >= 0 ? '#4ade80' : '#ef4444',
          }}>
            {cumulative >= 0 ? '+' : ''}{cumulative.toFixed(1)}
          </span>
        )}
      </div>

      {/* ── Hint ──────────────────────────────────────────────────── */}
      {entries.length > 0 && (
        <div style={{
          padding:    '3px 10px 3px 16px',
          borderBottom: '0.5px solid #1f1f1f',
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize:   '7px',
          color:      '#2a2a2a',
          flexShrink: 0,
        }}>
          click entry to expand ↓ · drag left edge to resize
        </div>
      )}

      {/* ── Entries ───────────────────────────────────────────────── */}
      <div
        ref={scrollRef}
        style={{ flex: 1, overflowY: 'auto', paddingLeft: '6px' }}
      >
        {entries.length === 0 ? (
          <div style={{
            padding:    '14px 10px',
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize:   '9px',
            color:      '#2a2a2a',
            lineHeight: 1.6,
          }}>
            No actions yet.{'\n'}Start an episode.
          </div>
        ) : (
          entries.map(entry => (
            <LogEntry
              key={entry.id}
              entry={entry}
              isNew={entry.id === newStepId}
            />
          ))
        )}
      </div>

      {/* ── Keyframes ─────────────────────────────────────────────── */}
      <style>{`
        @keyframes kaizen-fadein {
          from { opacity: 0; transform: translateY(-3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
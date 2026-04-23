import { useState, useEffect, useRef } from 'react';

/**
 * ActionLog
 * Right column — scrollable history of every action taken this episode.
 *
 * Design rules
 * ------------
 * - New entries fade in with kaizen-fadein (opacity 0 + translateY(-3px) → normal, 0.3s)
 * - Monospace font throughout
 * - Colour-coded by action type and outcome
 * - Auto-scrolls to the latest entry
 * - Resets when episode number changes
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
    case 'kill_process':
      return `kill pid ${action.pid}`;
    case 'thermal_mitigation':
      return `thermal → ${action.strategy}`;
    case 'allocate_memory':
      return `free ${action.mb_to_free}MB pid ${action.target_pid}`;
    case 'prioritize_task':
      return `priority ${action.priority} pid ${action.pid}`;
    case 'inspect_logs':
      return action.pid != null ? `logs pid ${action.pid}` : 'system logs';
    case 'list_processes':
      return 'list processes';
    case 'wait':
      return `wait`;
    case 'parse_error':
      return 'parse error';
    default:
      return action.tool_name;
  }
}

// ---------------------------------------------------------------------------
// Single log entry
// ---------------------------------------------------------------------------
function LogEntry({ entry, isNew }) {
  const colour = actionColour(entry.action?.tool_name);
  const icon   = actionIcon(entry.action?.tool_name);
  const desc   = shortDesc(entry.action);
  const success = entry.result?.success;

  return (
    <div style={{
      padding: '6px 10px',
      borderBottom: '0.5px solid #1f1f1f',
      animation: isNew ? 'kaizen-fadein 0.3s ease forwards' : 'none',
      backgroundColor: entry.action?.tool_name === 'kill_process' && success
        ? 'rgba(239,68,68,0.04)'
        : 'transparent',
    }}>
      {/* Step + tool row */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '2px',
      }}>
        {/* Step badge */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '8px',
          color: '#444444',
          minWidth: '20px',
        }}>
          s{entry.step}
        </span>

        {/* Icon */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '10px',
          color: colour,
          width: '10px',
          textAlign: 'center',
          flexShrink: 0,
        }}>
          {icon}
        </span>

        {/* Action description */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '10px',
          color: colour,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          flex: 1,
        }}>
          {desc}
        </span>

        {/* Reward delta */}
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: entry.reward > 0 ? '#4ade80' : entry.reward < 0 ? '#ef4444' : '#444444',
          flexShrink: 0,
          minWidth: '36px',
          textAlign: 'right',
        }}>
          {entry.reward > 0 ? '+' : ''}{entry.reward.toFixed(2)}
        </span>
      </div>

      {/* Result message */}
      {entry.result?.message && (
        <div style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '8px',
          color: success === false ? '#ef4444' : '#444444',
          paddingLeft: '36px',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          opacity: 0.8,
        }}>
          {entry.result.message.split('\n')[0]}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ActionLog({ state }) {
  const [entries, setEntries] = useState([]);
  const [newStepId, setNewStepId] = useState(null);
  const lastEpisodeRef = useRef(null);
  const lastStepRef    = useRef(null);
  const scrollRef      = useRef(null);

  useEffect(() => {
    if (!state) return;

    const episode = state.episode;
    const step    = state.step;

    // Reset log on new episode
    if (episode !== lastEpisodeRef.current) {
      setEntries([]);
      lastEpisodeRef.current = episode;
      lastStepRef.current    = null;
    }

    // Skip if same step as last update
    if (step === lastStepRef.current) return;
    lastStepRef.current = step;

    const entryId = `${episode}-${step}`;
    setNewStepId(entryId);

    setEntries(prev => {
      // Avoid duplicate entries
      if (prev.some(e => e.id === entryId)) return prev;
      return [
        ...prev,
        {
          id:      entryId,
          step:    step,
          episode: episode,
          action:  state.action,
          result:  state.action_result,
          reward:  state.reward ?? 0,
        },
      ];
    });

    // Clear new-entry highlight after animation completes
    const t = setTimeout(() => setNewStepId(null), 400);
    return () => clearTimeout(t);
  }, [state]);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const cumulative = entries.reduce((sum, e) => sum + e.reward, 0);

  return (
    <div style={{
      width: '160px',
      flexShrink: 0,
      borderLeft: '0.5px solid #1f1f1f',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '7px 10px',
        borderBottom: '0.5px solid #1f1f1f',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#444444',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Action Log
        </span>
        {entries.length > 0 && (
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: cumulative >= 0 ? '#4ade80' : '#ef4444',
          }}>
            {cumulative >= 0 ? '+' : ''}{cumulative.toFixed(1)}
          </span>
        )}
      </div>

      {/* Entries */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflow: 'auto',
        }}
      >
        {entries.length === 0 ? (
          <div style={{
            padding: '14px 10px',
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: '#2a2a2a',
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

      {/* Keyframes */}
      <style>{`
        @keyframes kaizen-fadein {
          from { opacity: 0; transform: translateY(-3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

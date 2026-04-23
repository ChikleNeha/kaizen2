import { useState, useEffect, useRef } from 'react';

/**
 * ReasoningPanel
 * Center column — observation summary, typewriter chain-of-thought,
 * and action pill showing the agent's last decision.
 *
 * Spec requirements (Section 5.5)
 * --------------------------------
 * - Three sections: Observation (muted), Chain-of-thought (typewriter), Action pill
 * - Typewriter: useEffect reveals text char-by-char at 18ms per character
 * - New state → reset typewriter and replay from beginning
 * - Action pill: bg #0a1a0a, border 0.5px solid #1a3a1a, text #4ade80
 * - blink keyframe for text cursor: opacity 1→0, 1s
 */

// ---------------------------------------------------------------------------
// Typewriter hook
// ---------------------------------------------------------------------------
function useTypewriter(text, speed = 18) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);
  const timerRef = useRef(null);
  const indexRef = useRef(0);

  useEffect(() => {
    // Reset on new text
    setDisplayed('');
    setDone(false);
    indexRef.current = 0;

    if (!text) {
      setDone(true);
      return;
    }

    function tick() {
      indexRef.current += 1;
      setDisplayed(text.slice(0, indexRef.current));
      if (indexRef.current < text.length) {
        timerRef.current = setTimeout(tick, speed);
      } else {
        setDone(true);
      }
    }

    timerRef.current = setTimeout(tick, speed);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [text, speed]);

  return { displayed, done };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatActionPill(action) {
  if (!action || !action.tool_name) return null;

  const name = action.tool_name;
  const parts = [name];

  if (action.pid != null)      parts.push(`pid=${action.pid}`);
  if (action.target_pid != null) parts.push(`pid=${action.target_pid}`);
  if (action.strategy != null) parts.push(`strategy=${action.strategy}`);
  if (action.priority != null) parts.push(`priority=${action.priority}`);
  if (action.mb_to_free != null) parts.push(`mb=${action.mb_to_free}`);
  if (name === 'wait' && action.reason) {
    parts.push(`"${action.reason.slice(0, 40)}${action.reason.length > 40 ? '…' : ''}"`);
  }

  return parts.join('  ');
}

function formatObsSummary(obs) {
  if (!obs) return 'Waiting for system state…';
  const chaos = obs.active_chaos
    ? `CHAOS: ${obs.active_chaos.toUpperCase()}`
    : 'no active chaos';
  return (
    `step ${obs.step ?? '—'}  ·  ` +
    `cpu ${(obs.cpu_percent ?? 0).toFixed(1)}%  ·  ` +
    `ram ${(obs.ram_percent ?? 0).toFixed(1)}%  ·  ` +
    `thermal ${(obs.thermal_celsius ?? 0).toFixed(1)}°C  ·  ` +
    chaos
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ReasoningPanel({ state }) {
  const obs            = state?.obs ?? null;
  const rawReasoning   = state?.agent_reasoning ?? '';
  const action         = state?.action ?? null;
  const actionResult   = state?.action_result ?? null;

  const { displayed: typedReasoning, done: typingDone } = useTypewriter(rawReasoning, 18);

  const actionLabel = formatActionPill(action);
  const obsSummary  = formatObsSummary(obs);

  const resultSuccess = actionResult?.success;

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      minWidth: 0,
    }}>

      {/* ── Observation summary bar ─────────────────────────────────── */}
      <div style={{
        padding: '7px 14px',
        borderBottom: '0.5px solid #1f1f1f',
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '9px',
        color: obs?.active_chaos ? '#f59e0b' : '#888888',
        letterSpacing: '0.04em',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        transition: 'color 0.5s ease',
        flexShrink: 0,
      }}>
        {obsSummary}
      </div>

      {/* ── Chain-of-thought typewriter ──────────────────────────────── */}
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '12px 14px',
        backgroundColor: '#111111',
      }}>
        {rawReasoning ? (
          <div style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '10px',
            color: '#888888',
            lineHeight: '1.7',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {typedReasoning}
            {/* Blinking cursor while typing */}
            {!typingDone && (
              <span style={{
                display: 'inline-block',
                width: '6px',
                height: '11px',
                backgroundColor: '#4ade80',
                marginLeft: '1px',
                verticalAlign: 'text-bottom',
                animation: 'kaizen-blink 1s step-end infinite',
              }} />
            )}
          </div>
        ) : (
          <div style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '10px',
            color: '#2a2a2a',
            lineHeight: '1.7',
          }}>
            {state
              ? 'Agent is thinking…'
              : 'Waiting for episode to start. Click Start Episode in the top bar.'}
          </div>
        )}
      </div>

      {/* ── Action pill ─────────────────────────────────────────────── */}
      {actionLabel && (
        <div style={{
          padding: '8px 14px',
          borderTop: '0.5px solid #1f1f1f',
          flexShrink: 0,
          animation: 'kaizen-fadein 0.3s ease forwards',
        }}>
          {/* Tool call pill */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: '#0a1a0a',
            border: '0.5px solid #1a3a1a',
            borderRadius: '4px',
            padding: '5px 10px',
            maxWidth: '100%',
          }}>
            <span style={{
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '10px',
              color: '#4ade80',
              whiteSpace: 'pre',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}>
              ▶ {actionLabel}
            </span>
          </div>

          {/* Result line */}
          {actionResult?.message && (
            <div style={{
              marginTop: '5px',
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '9px',
              color: resultSuccess === false ? '#ef4444' : '#4ade80',
              opacity: 0.8,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              transition: 'color 0.3s ease',
            }}>
              {resultSuccess === false ? '✗' : '✓'} {actionResult.message}
            </div>
          )}
        </div>
      )}

      {/* Keyframes */}
      <style>{`
        @keyframes kaizen-blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0; }
        }
        @keyframes kaizen-fadein {
          from { opacity: 0; transform: translateY(-3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

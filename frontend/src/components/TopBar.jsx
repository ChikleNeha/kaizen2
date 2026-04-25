import { useState, useCallback } from 'react';

/**
 * TopBar
 * Full-width top bar — logo, live status dot, episode badge, step counter,
 * model info, and "Start Episode" button.
 *
 * Design rules
 * ------------
 * - Height: 38px
 * - 0.5px bottom border #1f1f1f
 * - Monospace font only
 * - Status dot: kaizen-pulse animation (opacity 1→0.3→1, 2s)
 * - No shadows, no rounded corners > 6px
 * - Start button: calls POST /start_episode on the backend
 */

const WS_BASE = import.meta.env.VITE_API_URL
  ?? (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

// ---------------------------------------------------------------------------
// Status dot colours
// ---------------------------------------------------------------------------
const STATUS_COLOURS = {
  idle:     '#444444',
  loading:  '#f59e0b',
  running:  '#4ade80',
  done:     '#4ade80',
  error:    '#ef4444',
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function TopBar({ state, connected, lastEvent }) {
  const [requesting, setRequesting] = useState(false);
  const [apiError, setApiError]     = useState(null);

  // Derive status from ws state + lastEvent
  const wsStatus = lastEvent?.status ?? (connected ? 'idle' : 'disconnected');
  const isRunning = wsStatus === 'running' || wsStatus === 'loading';

  const dotColour = connected
    ? (STATUS_COLOURS[wsStatus] ?? '#888888')
    : '#ef4444';

  const dotPulse = wsStatus === 'running' || wsStatus === 'loading' || !connected;

  // Episode + step from live state
  const episode  = state?.episode  ?? lastEvent?.episode ?? 0;
  const step     = state?.step     ?? 0;
  const maxSteps = state?.max_steps ?? 10;

  // Model name from status endpoint (embedded in hello payload)
  const modelName = lastEvent?.model ?? 'Qwen2.5-3B';

  const handleStartEpisode = useCallback(async () => {
    if (requesting || isRunning) return;
    setRequesting(true);
    setApiError(null);
    try {
      const res = await fetch(`${WS_BASE}/start_episode`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'error') {
        setApiError(data.message);
      }
    } catch (err) {
      setApiError('Could not reach backend.');
    } finally {
      setRequesting(false);
    }
  }, [requesting, isRunning]);

  const buttonDisabled = requesting || isRunning;

  return (
    <div style={{
      height: '38px',
      borderBottom: '0.5px solid #1f1f1f',
      display: 'flex',
      alignItems: 'center',
      padding: '0 14px',
      gap: '16px',
      backgroundColor: '#0f0f0f',
      flexShrink: 0,
      zIndex: 10,
    }}>

      {/* Logo / project name */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '7px',
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '11px',
          fontWeight: '700',
          color: '#e2e2e2',
          letterSpacing: '0.06em',
        }}>
          KAIZEN
        </span>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#444444',
          letterSpacing: '0.04em',
        }}>
          OS
        </span>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '8px',
          color: '#2a2a2a',
        }}>
          /
        </span>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '8px',
          color: '#2a2a2a',
          letterSpacing: '0.04em',
        }}>
          agentic kernel
        </span>
      </div>

      {/* Vertical divider */}
      <Divider />

      {/* Status dot + label */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        flexShrink: 0,
      }}>
        <div style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: dotColour,
          animation: dotPulse ? 'kaizen-pulse 2s infinite' : 'none',
          transition: 'background-color 0.5s ease',
          flexShrink: 0,
        }} />
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: dotColour,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          transition: 'color 0.5s ease',
        }}>
          {connected ? wsStatus : 'disconnected'}
        </span>
      </div>

      {/* Vertical divider */}
      <Divider />

      {/* Episode badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#444444',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}>
          ep
        </span>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '11px',
          color: episode > 0 ? '#e2e2e2' : '#2a2a2a',
          fontWeight: '600',
        }}>
          {episode > 0 ? String(episode).padStart(2, '0') : '—'}
        </span>
      </div>

      {/* Step counter */}
      {state && (
        <>
          <Divider />
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
            <span style={{
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '9px',
              color: '#444444',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}>
              step
            </span>
            <span style={{
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '11px',
              color: '#e2e2e2',
              fontWeight: '600',
            }}>
              {step}
            </span>
            <span style={{
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '9px',
              color: '#444444',
            }}>
              / {maxSteps}
            </span>
          </div>
        </>
      )}

      {/* Chaos indicator */}
      {state?.obs?.active_chaos && (
        <>
          <Divider />
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: '#ef4444',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            animation: 'kaizen-flicker 0.9s infinite',
            flexShrink: 0,
          }}>
            ▲ {state.obs.active_chaos.replace('_', ' ')}
          </span>
        </>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Model badge */}
      <span style={{
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '8px',
        color: '#2a2a2a',
        letterSpacing: '0.04em',
        flexShrink: 0,
      }}>
        {modelName}
      </span>

      <Divider />

      {/* API error */}
      {apiError && (
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#ef4444',
          maxWidth: '140px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          flexShrink: 0,
        }}>
          {apiError}
        </span>
      )}

      {/* Start Episode button */}
      <button
        onClick={handleStartEpisode}
        disabled={buttonDisabled}
        style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          padding: '4px 12px',
          backgroundColor: buttonDisabled ? '#111111' : '#0a1a0a',
          border: `0.5px solid ${buttonDisabled ? '#1f1f1f' : '#1a3a1a'}`,
          borderRadius: '3px',
          color: buttonDisabled ? '#444444' : '#4ade80',
          cursor: buttonDisabled ? 'not-allowed' : 'pointer',
          transition: 'background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease',
          flexShrink: 0,
          outline: 'none',
        }}
        onMouseEnter={e => {
          if (!buttonDisabled) {
            e.currentTarget.style.backgroundColor = '#0f2a0f';
            e.currentTarget.style.borderColor = '#4ade80';
          }
        }}
        onMouseLeave={e => {
          if (!buttonDisabled) {
            e.currentTarget.style.backgroundColor = '#0a1a0a';
            e.currentTarget.style.borderColor = '#1a3a1a';
          }
        }}
      >
        {requesting ? 'starting…' : isRunning ? 'running…' : 'start episode'}
      </button>

      {/* Keyframes */}
      <style>{`
        @keyframes kaizen-pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
        @keyframes kaizen-flicker {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Divider helper
// ---------------------------------------------------------------------------
function Divider() {
  return (
    <div style={{
      width: '0.5px',
      height: '14px',
      backgroundColor: '#1f1f1f',
      flexShrink: 0,
    }} />
  );
}

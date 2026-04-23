import { useMemo } from 'react';

/**
 * StepsBar
 * Bottom bar — step progress dots + chaos event badge.
 *
 * Design rules
 * ------------
 * - One dot per max_steps (default 10)
 * - Completed steps: filled dot in step-appropriate colour
 * - Current step: pulsing dot
 * - Future steps: dim empty dot
 * - Chaos badge appears when active_chaos is set (amber/red)
 * - Terminated/truncated indicators at end
 * - 0.5px borders, monospace font, no shadows
 */

// ---------------------------------------------------------------------------
// Chaos badge colours
// ---------------------------------------------------------------------------
const CHAOS_COLOURS = {
  memory_leak:   { text: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.3)'  },
  cpu_hog:       { text: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.3)'  },
  thermal_spike: { text: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.3)' },
};

// ---------------------------------------------------------------------------
// Step dot colours — based on what happened at that step
// ---------------------------------------------------------------------------
function dotColour(stepIndex, currentStep, rewardHistory, chaosStep) {
  const stepNum = stepIndex + 1;

  if (stepNum > currentStep) return '#1f1f1f';          // future — dim
  if (stepNum === currentStep) return null;              // current — handled separately (pulse)

  // Completed step — colour by reward delta
  if (rewardHistory && rewardHistory.length >= stepNum) {
    const delta = stepNum === 1
      ? rewardHistory[0]
      : rewardHistory[stepNum - 1] - rewardHistory[stepNum - 2];
    if (delta > 1.0)  return '#4ade80';
    if (delta > 0)    return '#1a3a1a';
    if (delta < -2.0) return '#ef4444';
    if (delta < 0)    return '#3a1a1a';
    return '#2a2a2a';
  }

  // Chaos injection step
  if (stepNum === chaosStep) return '#f59e0b';

  return '#2a2a2a';
}

// ---------------------------------------------------------------------------
// Single step dot
// ---------------------------------------------------------------------------
function StepDot({ index, currentStep, rewardHistory, chaosStep, maxSteps }) {
  const stepNum   = index + 1;
  const isCurrent = stepNum === currentStep;
  const isFuture  = stepNum > currentStep;
  const colour    = dotColour(index, currentStep, rewardHistory, chaosStep);
  const isChaosStep = stepNum === chaosStep && stepNum <= currentStep;

  // Dot size
  const size = isCurrent ? 8 : 6;

  return (
    <div
      title={`Step ${stepNum}${isChaosStep ? ' (chaos)' : ''}`}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '14px',
        height: '14px',
        flexShrink: 0,
      }}
    >
      {/* Connecting line to next dot */}
      {index < maxSteps - 1 && (
        <div style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: 'calc(100% + 2px)',
          height: '0.5px',
          backgroundColor: isFuture ? '#1f1f1f' : '#2a2a2a',
          transform: 'translateY(-50%)',
          zIndex: 0,
        }} />
      )}

      {/* Dot */}
      <div style={{
        width:  `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        backgroundColor: isCurrent ? '#4ade80' : isFuture ? 'transparent' : colour,
        border: isFuture
          ? '0.5px solid #1f1f1f'
          : isCurrent
          ? '0.5px solid #4ade80'
          : `0.5px solid ${colour}`,
        flexShrink: 0,
        position: 'relative',
        zIndex: 1,
        animation: isCurrent ? 'kaizen-pulse 2s infinite' : 'none',
        transition: 'background-color 0.4s ease, border-color 0.4s ease, width 0.2s ease, height 0.2s ease',
      }} />

      {/* Chaos marker — small amber tick below dot */}
      {isChaosStep && (
        <div style={{
          position: 'absolute',
          bottom: '-1px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '2px',
          height: '2px',
          borderRadius: '50%',
          backgroundColor: '#f59e0b',
          zIndex: 2,
        }} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function StepsBar({ state }) {
  const step          = state?.step          ?? 0;
  const maxSteps      = state?.max_steps     ?? 10;
  const rewardHistory = state?.reward_history ?? [];
  const activeChaos   = state?.obs?.active_chaos ?? null;
  const terminated    = state?.terminated    ?? false;
  const truncated     = state?.truncated     ?? false;
  const episode       = state?.episode       ?? 0;

  // Chaos always injects at step 3
  const CHAOS_STEP = 3;

  const chaosInfo = activeChaos ? CHAOS_COLOURS[activeChaos] : null;

  // Status label
  const statusLabel = useMemo(() => {
    if (!state) return 'idle';
    if (terminated && !truncated) return 'resolved';
    if (truncated) return 'truncated';
    if (step > 0) return 'running';
    return 'ready';
  }, [state, terminated, truncated, step]);

  const statusColour = {
    idle:      '#444444',
    ready:     '#888888',
    running:   '#4ade80',
    resolved:  '#4ade80',
    truncated: '#f59e0b',
  }[statusLabel] ?? '#888888';

  return (
    <div style={{
      height: '36px',
      borderTop: '0.5px solid #1f1f1f',
      display: 'flex',
      alignItems: 'center',
      padding: '0 14px',
      gap: '12px',
      backgroundColor: '#0f0f0f',
      flexShrink: 0,
    }}>

      {/* Episode badge */}
      <span style={{
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '9px',
        color: '#444444',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        whiteSpace: 'nowrap',
        flexShrink: 0,
      }}>
        ep {episode > 0 ? String(episode).padStart(2, '0') : '—'}
      </span>

      {/* Divider */}
      <div style={{ width: '0.5px', height: '14px', backgroundColor: '#1f1f1f', flexShrink: 0 }} />

      {/* Step dots */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0',
        flex: 1,
        justifyContent: 'center',
        maxWidth: '200px',
      }}>
        {Array.from({ length: maxSteps }).map((_, i) => (
          <StepDot
            key={i}
            index={i}
            currentStep={step}
            rewardHistory={rewardHistory}
            chaosStep={CHAOS_STEP}
            maxSteps={maxSteps}
          />
        ))}
      </div>

      {/* Chaos badge */}
      {activeChaos && chaosInfo ? (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          padding: '2px 8px',
          backgroundColor: chaosInfo.bg,
          border: `0.5px solid ${chaosInfo.border}`,
          borderRadius: '3px',
          animation: 'kaizen-fadein 0.3s ease',
          flexShrink: 0,
        }}>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '8px',
            color: chaosInfo.text,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            animation: 'kaizen-flicker 0.9s infinite',
          }}>
            ▲ {activeChaos.replace('_', ' ')}
          </span>
        </div>
      ) : (
        /* Placeholder to prevent layout shift when chaos resolves */
        <div style={{ width: '80px', flexShrink: 0 }} />
      )}

      {/* Divider */}
      <div style={{ width: '0.5px', height: '14px', backgroundColor: '#1f1f1f', flexShrink: 0 }} />

      {/* Status */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '5px',
        flexShrink: 0,
      }}>
        <div style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          backgroundColor: statusColour,
          animation: statusLabel === 'running' ? 'kaizen-pulse 2s infinite' : 'none',
          transition: 'background-color 0.5s ease',
        }} />
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: statusColour,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          transition: 'color 0.5s ease',
        }}>
          {statusLabel}
        </span>
      </div>

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
        @keyframes kaizen-fadein {
          from { opacity: 0; transform: translateY(-3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

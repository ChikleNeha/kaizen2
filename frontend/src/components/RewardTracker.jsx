import { useMemo } from 'react';

/**
 * RewardTracker
 * Right column — large cumulative reward number, step delta, mini bar chart.
 *
 * Spec requirements (Section 5.6)
 * --------------------------------
 * - Large reward number: 26px, #4ade80, bold
 * - Delta line: "+X.X this step" in smaller green text
 * - Mini bar chart: flex row of bars, heights proportional to reward_history
 * - Active (latest) bar: #4ade80. Previous bars: #1a3a1a
 * - Bars animate height change with CSS transition
 * - No shadows, no rounded corners > 6px, 0.5px borders
 * - Monospace font only
 */

// ---------------------------------------------------------------------------
// Mini bar chart
// ---------------------------------------------------------------------------
function RewardBarChart({ history = [] }) {
  const BAR_MAX_H = 36;   // px — tallest possible bar
  const BAR_MIN_H = 2;    // px — floor so bars are always visible
  const BAR_W     = 8;    // px
  const BAR_GAP   = 2;    // px

  // Normalise: map reward_history values (cumulative) to bar heights.
  // We use step-level deltas for bar height so each bar shows per-step gain.
  const deltas = useMemo(() => {
    if (!history || history.length === 0) return [];
    const d = [];
    for (let i = 0; i < history.length; i++) {
      d.push(i === 0 ? history[0] : history[i] - history[i - 1]);
    }
    return d;
  }, [history]);

  const maxAbs = useMemo(() => {
    if (deltas.length === 0) return 1;
    return Math.max(1, Math.max(...deltas.map(Math.abs)));
  }, [deltas]);

  if (history.length === 0) {
    return (
      <div style={{
        height: `${BAR_MAX_H}px`,
        display: 'flex',
        alignItems: 'flex-end',
        gap: `${BAR_GAP}px`,
      }}>
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} style={{
            width: `${BAR_W}px`,
            height: `${BAR_MIN_H}px`,
            backgroundColor: '#1a2a1a',
            borderRadius: '1px',
            flexShrink: 0,
          }} />
        ))}
      </div>
    );
  }

  return (
    <div style={{
      height: `${BAR_MAX_H + 4}px`,
      display: 'flex',
      alignItems: 'flex-end',
      gap: `${BAR_GAP}px`,
      overflow: 'hidden',
    }}>
      {deltas.map((delta, i) => {
        const isActive  = i === deltas.length - 1;
        const isNeg     = delta < 0;
        const heightPct = Math.abs(delta) / maxAbs;
        const h = Math.max(BAR_MIN_H, Math.round(heightPct * BAR_MAX_H));

        let bg;
        if (isActive) {
          bg = isNeg ? '#ef4444' : '#4ade80';
        } else {
          bg = isNeg ? '#3a1a1a' : '#1a3a1a';
        }

        return (
          <div
            key={i}
            title={`Step ${i + 1}: ${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`}
            style={{
              width: `${BAR_W}px`,
              height: `${h}px`,
              backgroundColor: bg,
              borderRadius: '1px',
              flexShrink: 0,
              transition: 'height 0.4s ease, background-color 0.4s ease',
            }}
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function RewardTracker({ state }) {
  const cumulative    = state?.cumulative_reward ?? 0;
  const stepReward    = state?.reward ?? 0;
  const rewardHistory = state?.reward_history ?? [];
  const episode       = state?.episode ?? 0;
  const step          = state?.step ?? 0;
  const maxSteps      = state?.max_steps ?? 10;

  const deltaSign  = stepReward >= 0 ? '+' : '';
  const deltaColour = stepReward > 0
    ? '#4ade80'
    : stepReward < 0
    ? '#ef4444'
    : '#444444';

  const cumulativeColour = cumulative >= 0 ? '#4ade80' : '#ef4444';

  // Episode-level stats
  const episodeAvg = rewardHistory.length > 0
    ? (cumulative / rewardHistory.length).toFixed(2)
    : '—';

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
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#444444',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Reward
        </span>
      </div>

      {/* Main content */}
      <div style={{
        padding: '12px 10px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        flex: 1,
      }}>

        {/* Cumulative reward — large number */}
        <div>
          <div style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: '#444444',
            marginBottom: '3px',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>
            cumulative
          </div>
          <div style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '26px',
            fontWeight: '700',
            color: cumulativeColour,
            lineHeight: 1,
            transition: 'color 0.5s ease',
            letterSpacing: '-0.02em',
          }}>
            {cumulative >= 0 ? '+' : ''}{cumulative.toFixed(1)}
          </div>

          {/* Step delta */}
          {state && (
            <div style={{
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '9px',
              color: deltaColour,
              marginTop: '4px',
              transition: 'color 0.4s ease',
            }}>
              {deltaSign}{stepReward.toFixed(2)} this step
            </div>
          )}
        </div>

        {/* Divider */}
        <div style={{ borderTop: '0.5px solid #1f1f1f' }} />

        {/* Bar chart */}
        <div>
          <div style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: '#444444',
            marginBottom: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>
            per step
          </div>
          <RewardBarChart history={rewardHistory} />
        </div>

        {/* Divider */}
        <div style={{ borderTop: '0.5px solid #1f1f1f' }} />

        {/* Stats grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <StatRow label="episode" value={episode > 0 ? `#${episode}` : '—'} />
          <StatRow
            label="step"
            value={state ? `${step} / ${maxSteps}` : '—'}
          />
          <StatRow
            label="avg / step"
            value={episodeAvg !== '—' ? `${episodeAvg > 0 ? '+' : ''}${episodeAvg}` : '—'}
            colour={
              episodeAvg !== '—'
                ? parseFloat(episodeAvg) >= 0 ? '#4ade80' : '#ef4444'
                : '#444444'
            }
          />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stat row helper
// ---------------------------------------------------------------------------
function StatRow({ label, value, colour = '#e2e2e2' }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
    }}>
      <span style={{
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '9px',
        color: '#444444',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '10px',
        color: colour,
        transition: 'color 0.4s ease',
      }}>
        {value}
      </span>
    </div>
  );
}

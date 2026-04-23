import { useMemo } from 'react';

/**
 * VitalsPanel
 * Left column — CPU, RAM, Thermal metric bars + scrollable process list.
 *
 * Design rules (spec Section 5.1)
 * --------------------------------
 * - Monospace font only
 * - No shadows, no rounded corners > 6px
 * - 0.5px solid #1f1f1f borders
 * - Bars animate width with CSS transition: width 1s ease
 * - Value colours transition with: transition: color 0.5s ease
 * - No gradients on backgrounds
 * - green=#4ade80 (healthy), amber=#f59e0b (warning), red=#ef4444 (critical)
 * - indigo=#818cf8 for protected processes
 */

// ---------------------------------------------------------------------------
// Colour thresholds
// ---------------------------------------------------------------------------
function metricColour(value, warn = 60, crit = 85) {
  if (value >= crit) return '#ef4444';
  if (value >= warn) return '#f59e0b';
  return '#4ade80';
}

function thermalColour(value) {
  if (value >= 85) return '#ef4444';
  if (value >= 70) return '#f59e0b';
  return '#4ade80';
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricBar({ label, value, max = 100, colourFn = metricColour, unit = '%' }) {
  const pct = Math.min(100, (value / max) * 100);
  const colour = colourFn(value);

  return (
    <div style={{ marginBottom: '10px' }}>
      {/* Label row */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: '4px',
      }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#888888',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}>
          {label}
        </span>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '12px',
          color: colour,
          transition: 'color 0.5s ease',
          fontWeight: '600',
        }}>
          {typeof value === 'number' ? value.toFixed(1) : '—'}{unit}
        </span>
      </div>

      {/* Track */}
      <div style={{
        width: '100%',
        height: '3px',
        backgroundColor: '#1a1a1a',
        border: '0.5px solid #1f1f1f',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        {/* Fill */}
        <div style={{
          width: `${pct}%`,
          height: '100%',
          backgroundColor: colour,
          transition: 'width 1s ease, background-color 0.5s ease',
          borderRadius: '2px',
        }} />
      </div>
    </div>
  );
}

function ProcessRow({ proc, rank }) {
  const cpuColour = proc.is_protected
    ? '#818cf8'
    : metricColour(proc.cpu_percent, 30, 60);

  const isBad = !proc.is_protected && proc.cpu_percent > 60;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '18px 1fr 42px',
      alignItems: 'center',
      gap: '4px',
      padding: '4px 6px',
      borderBottom: '0.5px solid #1f1f1f',
      backgroundColor: isBad ? 'rgba(239,68,68,0.04)' : 'transparent',
      transition: 'background-color 0.5s ease',
    }}>
      {/* Rank */}
      <span style={{
        fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
        fontSize: '9px',
        color: '#444444',
        textAlign: 'right',
      }}>
        {rank}
      </span>

      {/* Name + PID */}
      <div style={{ overflow: 'hidden' }}>
        <div style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '10px',
          color: proc.is_protected ? '#818cf8' : '#e2e2e2',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}>
          {proc.is_protected && (
            <span style={{ color: '#818cf8', fontSize: '7px' }}>●</span>
          )}
          {isBad && (
            <span style={{
              color: '#ef4444',
              fontSize: '7px',
              animation: 'kaizen-flicker 0.9s infinite',
            }}>▲</span>
          )}
          {proc.name}
        </div>
        <div style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '8px',
          color: '#444444',
        }}>
          pid {proc.pid} · {proc.memory_mb.toFixed(0)} MB
        </div>
      </div>

      {/* CPU% */}
      <div style={{ textAlign: 'right' }}>
        <span style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '10px',
          color: cpuColour,
          transition: 'color 0.5s ease',
          fontWeight: isBad ? '700' : '400',
          animation: isBad ? 'kaizen-flicker 0.9s infinite' : 'none',
        }}>
          {proc.cpu_percent.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function VitalsPanel({ obs }) {
  const processList = useMemo(
    () => (obs?.process_list ?? []).slice().sort((a, b) => b.cpu_percent - a.cpu_percent),
    [obs]
  );

  const cpu     = obs?.cpu_percent     ?? 0;
  const ram     = obs?.ram_percent     ?? 0;
  const thermal = obs?.thermal_celsius ?? 0;

  return (
    <div style={{
      width: '220px',
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      gap: '0',
      borderRight: '0.5px solid #1f1f1f',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Metrics section */}
      <div style={{
        padding: '12px 14px 10px',
        borderBottom: '0.5px solid #1f1f1f',
      }}>
        <div style={{
          fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#444444',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          marginBottom: '10px',
        }}>
          System Vitals
        </div>

        <MetricBar
          label="CPU"
          value={cpu}
          colourFn={metricColour}
        />
        <MetricBar
          label="RAM"
          value={ram}
          colourFn={metricColour}
        />
        <MetricBar
          label="Thermal"
          value={thermal}
          max={110}
          colourFn={thermalColour}
          unit="°C"
        />

        {/* Uptime */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '8px',
          paddingTop: '8px',
          borderTop: '0.5px solid #1f1f1f',
        }}>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '9px',
            color: '#888888',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>Uptime</span>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '10px',
            color: '#e2e2e2',
          }}>
            {obs?.uptime_seconds != null
              ? `${obs.uptime_seconds.toFixed(0)}s`
              : '—'}
          </span>
        </div>
      </div>

      {/* Process list section */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '18px 1fr 42px',
          gap: '4px',
          padding: '5px 6px',
          borderBottom: '0.5px solid #1f1f1f',
          backgroundColor: '#111111',
        }}>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '8px',
            color: '#444444',
            textAlign: 'right',
          }}>#</span>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '8px',
            color: '#444444',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}>Process</span>
          <span style={{
            fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
            fontSize: '8px',
            color: '#444444',
            textAlign: 'right',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}>CPU</span>
        </div>

        {/* Rows */}
        <div style={{ overflow: 'auto', flex: 1 }}>
          {processList.length === 0 ? (
            <div style={{
              padding: '16px',
              fontFamily: "ui-monospace, 'JetBrains Mono', monospace",
              fontSize: '10px',
              color: '#444444',
              textAlign: 'center',
            }}>
              No process data
            </div>
          ) : (
            processList.map((proc, i) => (
              <ProcessRow key={proc.pid} proc={proc} rank={i + 1} />
            ))
          )}
        </div>
      </div>

      {/* Keyframe styles injected once */}
      <style>{`
        @keyframes kaizen-flicker {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

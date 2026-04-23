import { useMemo } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import TopBar         from './components/TopBar';
import VitalsPanel    from './components/VitalsPanel';
import ProcessGraph   from './components/ProcessGraph';
import ReasoningPanel from './components/ReasoningPanel';
import ActionLog      from './components/ActionLog';
import RewardTracker  from './components/RewardTracker';
import StepsBar       from './components/StepsBar';

/**
 * App — root dashboard layout
 *
 * Layout (spec Section 5.2)
 * ─────────────────────────────────────────────────────────
 * ┌─────────────────────────────────────────────────────┐
 * │  TOPBAR (38px)                                      │
 * ├──────────┬──────────────────────────┬───────────────┤
 * │  Left    │   Center                 │  Right        │
 * │  220px   │   flex:1                 │  160px        │
 * │          │                          │               │
 * │ Vitals   │  ProcessGraph (220px h)  │  Reward       │
 * │ Metrics  ├──────────────────────────┤  Tracker      │
 * │          │  ReasoningPanel          │               │
 * │ Process  │                          │  Action Log   │
 * │ List     │                          │               │
 * ├──────────┴──────────────────────────┴───────────────┤
 * │  StepsBar (36px)                                    │
 * └─────────────────────────────────────────────────────┘
 *
 * Full-viewport, no scrolling on the outer container.
 * Right column splits vertically: RewardTracker top, ActionLog below.
 */

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws';

export default function App() {
  const { state, connected, lastEvent } = useWebSocket(WS_URL);

  // Process list extracted for ProcessGraph — memoised to avoid re-renders
  const processList = useMemo(
    () => state?.obs?.process_list ?? [],
    [state?.obs?.process_list]
  );

  return (
    <div style={{
      width:           '100vw',
      height:          '100vh',
      backgroundColor: '#0f0f0f',
      display:         'flex',
      flexDirection:   'column',
      overflow:        'hidden',
      fontFamily:      "ui-monospace, 'JetBrains Mono', 'Fira Code', monospace",
    }}>

      {/* ── TopBar ──────────────────────────────────────────────── */}
      <TopBar
        state={state}
        connected={connected}
        lastEvent={lastEvent}
      />

      {/* ── Main area ───────────────────────────────────────────── */}
      <div style={{
        flex:     1,
        display:  'flex',
        overflow: 'hidden',
        minHeight: 0,
      }}>

        {/* ── Left column: VitalsPanel ──────────────────────────── */}
        <VitalsPanel obs={state?.obs ?? null} />

        {/* ── Center column ─────────────────────────────────────── */}
        <div style={{
          flex:          1,
          display:       'flex',
          flexDirection: 'column',
          overflow:      'hidden',
          minWidth:      0,
        }}>
          {/* ProcessGraph — fixed height */}
          <div style={{
            height:       '220px',
            flexShrink:   0,
            borderBottom: '0.5px solid #1f1f1f',
            backgroundColor: '#111111',
            overflow:     'hidden',
          }}>
            <ProcessGraph
              processList={processList}
              height={220}
            />
          </div>

          {/* ReasoningPanel — fills remaining center height */}
          <ReasoningPanel state={state} />
        </div>

        {/* ── Right column ──────────────────────────────────────── */}
        <div style={{
          width:         '160px',
          flexShrink:    0,
          display:       'flex',
          flexDirection: 'column',
          overflow:      'hidden',
          borderLeft:    '0.5px solid #1f1f1f',
        }}>
          {/* RewardTracker — top half */}
          <div style={{
            flex:         '0 0 auto',
            borderBottom: '0.5px solid #1f1f1f',
          }}>
            <RewardTracker state={state} />
          </div>

          {/* ActionLog — fills remaining right column height */}
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <ActionLog state={state} />
          </div>
        </div>
      </div>

      {/* ── StepsBar ────────────────────────────────────────────── */}
      <StepsBar state={state} />

      {/* ── Global base styles ──────────────────────────────────── */}
      <style>{`
        *, *::before, *::after {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        html, body, #root {
          width: 100%;
          height: 100%;
          overflow: hidden;
          background-color: #0f0f0f;
        }

        /* Minimal scrollbar styling for panels that allow overflow:auto */
        ::-webkit-scrollbar {
          width: 3px;
          height: 3px;
        }
        ::-webkit-scrollbar-track {
          background: #111111;
        }
        ::-webkit-scrollbar-thumb {
          background: #1f1f1f;
          border-radius: 2px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #2a2a2a;
        }
      `}</style>
    </div>
  );
}

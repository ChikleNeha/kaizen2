import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * useWebSocket
 * Connects to the Kaizen OS backend WebSocket and returns live state.
 *
 * Features
 * --------
 * - Auto-reconnects every 2 seconds on disconnect
 * - Parses only "state_update" messages into state
 * - Tracks connection status separately
 * - Cleans up the socket on component unmount
 *
 * @param {string} url  WebSocket URL, e.g. "ws://localhost:8000/ws"
 * @returns {{ state: object|null, connected: boolean, lastEvent: object|null }}
 */
export function useWebSocket(url) {
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState(null);   // raw last message for debugging

  // Stable refs so the reconnect closure doesn't stale-capture
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    // Don't open a second socket if one is already connecting / open
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.CONNECTING ||
        wsRef.current.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    const ws = new WebSocket(url);

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setConnected(true);
      // Clear any pending reconnect timer
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      // Schedule reconnect only if still mounted
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, 2000);
    };

    ws.onerror = () => {
      // onerror is always followed by onclose — let onclose handle reconnect
      if (!mountedRef.current) return;
      setConnected(false);
    };

    ws.onmessage = (e) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(e.data);
        setLastEvent(data);
        if (data.type === 'state_update') {
          setState(data);
        }
        // "hello", "ping", "episode_done", "error", "status" messages
        // are stored in lastEvent but don't overwrite state
      } catch {
        // Malformed message — silently ignore
      }
    };

    wsRef.current = ws;
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      // Clear reconnect timer so it doesn't fire after unmount
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      // Close socket cleanly — code 1000 = normal closure
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close(1000, 'Component unmounted');
      }
    };
  }, [connect]);

  return { state, connected, lastEvent };
}

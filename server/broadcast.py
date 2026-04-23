"""
server/broadcast.py
WebSocket connection manager for the Kaizen OS dashboard.

All connected clients receive every state_update broadcast simultaneously.
Client disconnections are handled gracefully — dead sockets are pruned
without crashing the broadcast loop.

The module-level ``manager`` singleton is imported by both server/main.py
and environment/kaizen_env.py so both can share the same connection set.
"""

import asyncio
import json
import logging
from typing import Any, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages the set of active WebSocket connections.

    Thread-safety note
    ------------------
    FastAPI runs in a single asyncio event loop.  All WebSocket operations
    are awaited coroutines, so no explicit locking is needed — asyncio's
    cooperative scheduling ensures the active set is never mutated
    concurrently.  If you move to a multi-process deployment, replace the
    set with a Redis pub/sub channel.
    """

    def __init__(self) -> None:
        self.active: Set[WebSocket] = set()
        self._broadcast_count: int = 0
        self._connect_count: int = 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, ws: WebSocket) -> None:
        """
        Accept a new WebSocket connection and register it.

        Parameters
        ----------
        ws : WebSocket
            The incoming FastAPI WebSocket object.
        """
        await ws.accept()
        self.active.add(ws)
        self._connect_count += 1
        logger.info(
            f"[WS] Client connected. Active connections: {len(self.active)}"
        )

    def disconnect(self, ws: WebSocket) -> None:
        """
        Remove a WebSocket from the active set.

        Safe to call even if ws is not currently registered (no-op via discard).

        Parameters
        ----------
        ws : WebSocket
            The WebSocket to remove.
        """
        self.active.discard(ws)
        logger.info(
            f"[WS] Client disconnected. Active connections: {len(self.active)}"
        )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, data: dict[str, Any]) -> None:
        """
        Serialise ``data`` to JSON and send to all active clients.

        Dead connections (closed by client, network error, timeout) are
        collected during the iteration and pruned from the active set
        after the loop completes — never mid-iteration.

        Parameters
        ----------
        data : dict
            Must conform to the WebSocket state schema in Section 4.1.
            No validation is performed here; callers are responsible.
        """
        if not self.active:
            return

        message = json.dumps(data, default=str)  # default=str handles non-serialisable types
        dead: Set[WebSocket] = set()

        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception as exc:
                logger.debug(f"[WS] Send failed ({type(exc).__name__}), marking dead.")
                dead.add(ws)

        # Prune dead connections
        if dead:
            self.active -= dead
            logger.info(
                f"[WS] Pruned {len(dead)} dead connection(s). "
                f"Active: {len(self.active)}"
            )

        self._broadcast_count += 1

    async def broadcast_text(self, text: str) -> None:
        """
        Send a raw pre-serialised text string to all clients.

        Useful for sending keep-alive pings or error messages that are
        not full state_update payloads.

        Parameters
        ----------
        text : str
            Pre-formatted string to send verbatim.
        """
        if not self.active:
            return

        dead: Set[WebSocket] = set()

        for ws in self.active:
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)

        if dead:
            self.active -= dead

    # ------------------------------------------------------------------
    # Keep-alive ping
    # ------------------------------------------------------------------

    async def ping_all(self) -> None:
        """
        Send a JSON ping to all clients to detect stale connections early.

        Called periodically by the background task in server/main.py.
        Clients that do not respond will raise on the next send and be
        pruned then.
        """
        await self.broadcast_text(json.dumps({"type": "ping"}))

    # ------------------------------------------------------------------
    # Stats (used by GET /status endpoint)
    # ------------------------------------------------------------------

    @property
    def connection_count(self) -> int:
        """Current number of active connections."""
        return len(self.active)

    @property
    def total_broadcasts(self) -> int:
        """Total number of broadcast() calls since server start."""
        return self._broadcast_count

    @property
    def total_connections(self) -> int:
        """Total number of clients that have ever connected."""
        return self._connect_count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Imported by server/main.py and environment/kaizen_env.py.
# Both share the same in-process connection set.

manager = ConnectionManager()

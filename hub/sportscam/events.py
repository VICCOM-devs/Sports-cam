"""Per-match WebSocket fan-out. Tags, score, clock and camera status reach every viewer in <1 s."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, match_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs[match_id].add(ws)

    async def unsubscribe(self, match_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs[match_id].discard(ws)

    def count(self, match_id: str) -> int:
        return len(self._subs.get(match_id, ()))

    async def publish(self, match_id: str, kind: str, data: Any) -> None:
        msg = json.dumps({"type": kind, "data": data})
        dead: list[WebSocket] = []
        for ws in list(self._subs.get(match_id, ())):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subs[match_id].discard(ws)

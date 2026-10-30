"""Version gate (viccom-auto-update layer 1).

Fetches https://viccom.se/updates/sports-cam.json on start and every 30 minutes. The web apps
read the result from /api/version and show a nudge or a blocking screen. Offline or a broken
JSON never blocks: the hub keeps running and tries again later.

JSON shape (one entry per platform):
{
  "platforms": {
    "desktop": {"min_version_code": 1, "latest_version_code": 3, "force": false,
                "message_sv": "...", "message_en": "...",
                "store_url": "https://viccom.se/sports-cam"}
  }
}
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from . import version

log = logging.getLogger("sportscam.updates")


class UpdateChecker:
    def __init__(self, url: str, platform: str = "desktop", interval_s: int = 1800):
        self.url = url
        self.platform = platform
        self.interval_s = interval_s
        self.state: dict = {"status": "unknown", "checked_at": None}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if version.DISTRIBUTION != "direct":
            self.state = {"status": "store_managed", "checked_at": time.time()}
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await self.check()
            await asyncio.sleep(self.interval_s)

    async def check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
                r = await c.get(self.url, headers={"Cache-Control": "no-cache"})
                r.raise_for_status()
                entry = (r.json().get("platforms") or {}).get(self.platform) or {}
            self.state = self.evaluate(entry)
        except Exception as e:  # offline, DNS, bad JSON: never block
            log.info("update check skipped: %s", e)
            self.state = {"status": "unavailable", "checked_at": time.time()}
        return self.state

    @staticmethod
    def evaluate(entry: dict, build: int = version.BUILD_NUMBER) -> dict:
        min_code = int(entry.get("min_version_code", 0) or 0)
        latest = int(entry.get("latest_version_code", 0) or 0)
        force = bool(entry.get("force", False))
        if build < min_code or (force and build < latest):
            status = "blocked"
        elif build < latest:
            status = "nudge"
        else:
            status = "current"
        return {
            "status": status,
            "checked_at": time.time(),
            "latest_version_code": latest,
            "min_version_code": min_code,
            "message_sv": entry.get("message_sv", ""),
            "message_en": entry.get("message_en", ""),
            "store_url": entry.get("store_url", "https://viccom.se"),
        }

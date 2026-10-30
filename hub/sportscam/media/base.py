from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StreamInfo:
    path: str
    live: bool = False
    viewers: int = 0
    recording: bool = False
    tracks: list[str] = field(default_factory=list)


class MediaEngine:
    name = "base"

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def whip_publish(self, path: str, offer_sdp: str) -> tuple[str, str]:
        """Camera → hub. Returns (answer_sdp, session_id)."""
        raise NotImplementedError

    async def whip_delete(self, session_id: str) -> None: ...

    async def whep_play(self, path: str, offer_sdp: str) -> tuple[str, str]:
        """Hub → viewer. Returns (answer_sdp, session_id)."""
        raise NotImplementedError

    async def whep_delete(self, session_id: str) -> None: ...

    def streams(self) -> dict[str, StreamInfo]:
        return {}

    async def start_recording(self, path: str, file: Path) -> None: ...
    async def stop_recording(self, path: str) -> None: ...

    def frames(self, path: str, fps: float = 5.0):
        """Async generator of (wall_time, bgr_ndarray) decoded frames, rate-limited to fps.
        Used by the vision service and for calibration snapshots."""
        raise NotImplementedError

    async def snapshot(self, path: str):
        """One BGR frame or None."""
        async for _, frame in self.frames(path, fps=30):
            return frame
        return None

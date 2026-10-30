"""Built-in WebRTC engine on aiortc.

Each published path keeps its peer connection and raw tracks. Viewers get relayed copies via
MediaRelay; aiortc re-encodes per viewer, so this engine is for small setups and testing.
Recordings are written as MP4 with MediaRecorder.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaRelay

from .base import MediaEngine, StreamInfo

log = logging.getLogger("sportscam.media.pyrtc")


@dataclass
class _Publisher:
    path: str
    pc: RTCPeerConnection
    session_id: str
    tracks: dict[str, object] = field(default_factory=dict)   # kind -> MediaStreamTrack
    relay: MediaRelay = field(default_factory=MediaRelay)
    recorder: MediaRecorder | None = None
    record_file: Path | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Viewer:
    path: str
    pc: RTCPeerConnection
    session_id: str


class PyRtcEngine(MediaEngine):
    name = "pyrtc"

    def __init__(self) -> None:
        self._pubs: dict[str, _Publisher] = {}
        self._pub_by_session: dict[str, _Publisher] = {}
        self._viewers: dict[str, _Viewer] = {}

    async def stop(self) -> None:
        for v in list(self._viewers.values()):
            await v.pc.close()
        for p in list(self._pubs.values()):
            await self._close_publisher(p)

    # -- publish (WHIP) ------------------------------------------------------
    async def whip_publish(self, path: str, offer_sdp: str) -> tuple[str, str]:
        old = self._pubs.get(path)
        if old:  # a reconnecting camera replaces its old session
            await self._close_publisher(old)

        pc = RTCPeerConnection()
        pub = _Publisher(path=path, pc=pc, session_id=uuid.uuid4().hex)

        @pc.on("track")
        def on_track(track):
            pub.tracks[track.kind] = track
            log.info("publisher %s: %s track", path, track.kind)
            if "video" in pub.tracks:
                pub.ready.set()

            @track.on("ended")
            async def on_ended():
                pub.tracks.pop(track.kind, None)

        @pc.on("connectionstatechange")
        async def on_state():
            log.info("publisher %s: %s", path, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                if self._pubs.get(path) is pub:
                    await self._close_publisher(pub)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self._pubs[path] = pub
        self._pub_by_session[pub.session_id] = pub
        return pc.localDescription.sdp, pub.session_id

    async def whip_delete(self, session_id: str) -> None:
        pub = self._pub_by_session.get(session_id)
        if pub:
            await self._close_publisher(pub)

    async def _close_publisher(self, pub: _Publisher) -> None:
        self._pubs.pop(pub.path, None) if self._pubs.get(pub.path) is pub else None
        self._pub_by_session.pop(pub.session_id, None)
        if pub.recorder:
            try:
                await pub.recorder.stop()
            except Exception:
                log.exception("recorder stop failed for %s", pub.path)
            pub.recorder = None
        await pub.pc.close()

    # -- play (WHEP) ---------------------------------------------------------
    async def whep_play(self, path: str, offer_sdp: str) -> tuple[str, str]:
        pub = self._pubs.get(path)
        if not pub:
            raise KeyError(path)
        try:
            await asyncio.wait_for(pub.ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            raise KeyError(path)

        pc = RTCPeerConnection()
        viewer = _Viewer(path=path, pc=pc, session_id=uuid.uuid4().hex)
        for kind in ("video", "audio"):
            track = pub.tracks.get(kind)
            if track is not None:
                pc.addTrack(pub.relay.subscribe(track))

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                self._viewers.pop(viewer.session_id, None)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self._viewers[viewer.session_id] = viewer
        return pc.localDescription.sdp, viewer.session_id

    async def whep_delete(self, session_id: str) -> None:
        v = self._viewers.pop(session_id, None)
        if v:
            await v.pc.close()

    # -- info ----------------------------------------------------------------
    def streams(self) -> dict[str, StreamInfo]:
        out: dict[str, StreamInfo] = {}
        for path, pub in self._pubs.items():
            out[path] = StreamInfo(
                path=path,
                live="video" in pub.tracks,
                viewers=sum(1 for v in self._viewers.values() if v.path == path),
                recording=pub.recorder is not None,
                tracks=sorted(pub.tracks),
            )
        return out

    # -- recording -----------------------------------------------------------
    async def start_recording(self, path: str, file: Path) -> None:
        pub = self._pubs.get(path)
        if not pub or pub.recorder:
            return
        await asyncio.wait_for(pub.ready.wait(), timeout=5)
        file.parent.mkdir(parents=True, exist_ok=True)
        rec = MediaRecorder(str(file))
        for kind in ("video", "audio"):
            track = pub.tracks.get(kind)
            if track is not None:
                rec.addTrack(pub.relay.subscribe(track))
        await rec.start()
        pub.recorder, pub.record_file = rec, file
        log.info("recording %s -> %s", path, file)

    async def stop_recording(self, path: str) -> None:
        pub = self._pubs.get(path)
        if pub and pub.recorder:
            rec, pub.recorder = pub.recorder, None
            await rec.stop()

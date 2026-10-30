"""Built-in WebRTC engine on aiortc.

Each published path keeps its peer connection and raw tracks. Viewers get relayed copies via
MediaRelay; aiortc re-encodes per viewer, so this engine is for small setups and testing.
Recordings are written as MP4 with MediaRecorder.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

from .base import MediaEngine, StreamInfo

log = logging.getLogger("anysport.media.pyrtc")


class FfmpegRecorder:
    """Pipe decoded frames to an ffmpeg child process (x264 veryfast, Matroska). Robust: the file
    grows as frames arrive and stays playable even if the hub dies. Video only for now."""

    def __init__(self, track, file: Path, fps: int = 15):
        self.track = track
        self.file = file
        self.fps = fps
        self.proc: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task | None = None
        self.frames = 0

    async def start(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        import numpy as np  # noqa: F401  (frames come as ndarray)

        try:
            frame = await self.track.recv()
        except Exception:
            return
        w, h = frame.width, frame.height
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
               "-r", str(self.fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-g", str(self.fps * 2), str(self.file)]
        self.proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE,
                                                         stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        last_pts = None
        try:
            while True:
                if frame.width != w or frame.height != h:
                    frame = frame.reformat(width=w, height=h)
                data = frame.to_ndarray(format="yuv420p").tobytes()
                # constant frame rate: repeat frames if the camera is slower than fps, drop if faster
                t = frame.time if frame.time is not None else self.frames / self.fps
                if last_pts is None or t - last_pts >= 1 / self.fps * 0.5:
                    self.proc.stdin.write(data)
                    await self.proc.stdin.drain()
                    self.frames += 1
                    last_pts = t
                frame = await self.track.recv()
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
                self.proc.stdin.close()

    async def stop(self) -> None:
        self.track.stop()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
        if self.proc:
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()


@dataclass
class _Publisher:
    path: str
    pc: RTCPeerConnection
    session_id: str
    tracks: dict[str, object] = field(default_factory=dict)   # kind -> MediaStreamTrack
    relay: MediaRelay = field(default_factory=MediaRelay)
    recorder: FfmpegRecorder | None = None
    record_file: Path | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    player: MediaPlayer | None = None     # set for external sources


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
        if pub.player:
            for tr in (pub.player.video, pub.player.audio):
                if tr is not None:
                    tr.stop()
        await pub.pc.close()

    # -- external sources (rink camera over RTSP, capture card, file) ------------
    async def add_source(self, path: str, url: str) -> None:
        old = self._pubs.get(path)
        if old:
            await self._close_publisher(old)
        opts = {}
        if url.startswith("rtsp://"):
            opts = {"rtsp_transport": "tcp", "stimeout": "5000000"}
        loop = asyncio.get_running_loop()
        player = await loop.run_in_executor(None, lambda: MediaPlayer(url, options=opts, loop=url.endswith((".y4m", ".mp4", ".mkv", ".mov"))))
        if player.video is None:
            raise ValueError("source has no video")
        pub = _Publisher(path=path, pc=RTCPeerConnection(), session_id=uuid.uuid4().hex, player=player)
        pub.tracks["video"] = player.video
        if player.audio is not None:
            pub.tracks["audio"] = player.audio
        pub.ready.set()
        self._pubs[path] = pub
        self._pub_by_session[pub.session_id] = pub
        log.info("external source %s -> %s", url, path)

    async def remove_source(self, path: str) -> None:
        pub = self._pubs.get(path)
        if pub and pub.player:
            await self._close_publisher(pub)

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

    # -- frame tap -----------------------------------------------------------
    async def frames(self, path: str, fps: float = 5.0):
        pub = self._pubs.get(path)
        if not pub:
            return
        try:
            await asyncio.wait_for(pub.ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            return
        track = pub.relay.subscribe(pub.tracks["video"])
        min_dt = 1.0 / fps
        last = 0.0
        try:
            while self._pubs.get(path) is pub:
                frame = await track.recv()
                now = time.time()
                if now - last < min_dt:
                    continue
                last = now
                yield now, frame.to_ndarray(format="bgr24")
        except Exception:
            return
        finally:
            track.stop()

    # -- recording -----------------------------------------------------------
    async def start_recording(self, path: str, file: Path) -> None:
        pub = self._pubs.get(path)
        if not pub or pub.recorder:
            return
        await asyncio.wait_for(pub.ready.wait(), timeout=5)
        rec = FfmpegRecorder(pub.relay.subscribe(pub.tracks["video"]), file)
        await rec.start()
        pub.recorder, pub.record_file = rec, file
        log.info("recording %s -> %s", path, file)

    async def stop_recording(self, path: str) -> None:
        pub = self._pubs.get(path)
        if pub and pub.recorder:
            rec, pub.recorder = pub.recorder, None
            try:
                await rec.stop()
            except Exception as e:  # PyAV can refuse the final flush; the file is still playable
                log.warning("recorder stop for %s: %s", path, e)

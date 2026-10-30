"""MediaMTX engine: runs the official MediaMTX binary as a child process and proxies WHIP/WHEP.

First run downloads the pinned release for this OS from GitHub, verifies it against the
release's checksum file, and unpacks it into the hub data dir. Nothing is executed before
the checksum matches.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

from .base import MediaEngine, StreamInfo

log = logging.getLogger("sportscam.media.mediamtx")

MEDIAMTX_VERSION = "v1.15.3"
RELEASE_BASE = f"https://github.com/bluenviron/mediamtx/releases/download/{MEDIAMTX_VERSION}"

CONFIG_TEMPLATE = """\
logLevel: info
api: yes
apiAddress: 127.0.0.1:9997
metrics: no
rtsp: yes
rtspAddress: 127.0.0.1:8554
rtspTransports: [tcp]
rtmp: no
hls: no
srt: yes
srtAddress: :8890
webrtc: yes
webrtcAddress: :8889
webrtcLocalUDPAddress: :8189
webrtcIPsFromInterfaces: yes
webrtcICEServers2: []
record: no
pathDefaults:
  recordPath: {record_path}
  recordFormat: fmp4
  recordSegmentDuration: 60s
paths:
  all_others:
"""


def _asset_name() -> str:
    sysname = platform.system().lower()
    arch = platform.machine().lower()
    arch = {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(arch, arch)
    if sysname.startswith("win"):
        return f"mediamtx_{MEDIAMTX_VERSION}_windows_{arch}.zip"
    if sysname == "darwin":
        return f"mediamtx_{MEDIAMTX_VERSION}_darwin_{arch}.tar.gz"
    return f"mediamtx_{MEDIAMTX_VERSION}_linux_{arch}.tar.gz"


def binary_path(bin_dir: Path) -> Path:
    exe = "mediamtx.exe" if os.name == "nt" else "mediamtx"
    return bin_dir / MEDIAMTX_VERSION / exe


async def ensure_binary(bin_dir: Path) -> Path:
    """Download + verify + unpack MediaMTX if it is not already there."""
    target = binary_path(bin_dir)
    if target.exists():
        return target
    asset = _asset_name()
    dest_dir = target.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / asset
    log.info("downloading %s", asset)
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(f"{RELEASE_BASE}/{asset}")
        r.raise_for_status()
        archive.write_bytes(r.content)
        sums = await client.get(f"{RELEASE_BASE}/mediamtx_{MEDIAMTX_VERSION}_checksums.sha256")
        sums.raise_for_status()
    expected = None
    for line in sums.text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].strip("*") == asset:
            expected = parts[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected is None or actual != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"MediaMTX checksum mismatch for {asset}")
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extract(target.name, dest_dir)
    else:
        with tarfile.open(archive) as t:
            member = t.getmember(target.name)
            t.extract(member, dest_dir, filter="data")
    archive.unlink(missing_ok=True)
    target.chmod(0o755)
    return target


class MediaMtxEngine(MediaEngine):
    name = "mediamtx"

    def __init__(self, bin_dir: Path, recordings_dir: Path, api: str, webrtc: str):
        self.bin_dir = bin_dir
        self.recordings_dir = recordings_dir
        self.api = api.rstrip("/")
        self.webrtc = webrtc.rstrip("/")
        self.proc: subprocess.Popen | None = None
        self._client = httpx.AsyncClient(timeout=10)
        self._recording: set[str] = set()

    async def start(self) -> None:
        exe = await ensure_binary(self.bin_dir)
        cfg = self.bin_dir / "mediamtx.yml"
        record_path = (self.recordings_dir / "%path" / "%Y-%m-%d_%H-%M-%S-%f").as_posix()
        cfg.write_text(CONFIG_TEMPLATE.format(record_path=record_path))
        self.proc = subprocess.Popen([str(exe), str(cfg)], cwd=str(self.bin_dir))
        for _ in range(50):
            try:
                r = await self._client.get(f"{self.api}/v3/paths/list")
                if r.status_code == 200:
                    log.info("mediamtx %s ready", MEDIAMTX_VERSION)
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
        raise RuntimeError("MediaMTX did not start")

    async def stop(self) -> None:
        await self._client.aclose()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    async def _signal(self, path: str, kind: str, offer_sdp: str) -> tuple[str, str]:
        r = await self._client.post(
            f"{self.webrtc}/{path}/{kind}", content=offer_sdp,
            headers={"Content-Type": "application/sdp"},
        )
        if r.status_code == 404:
            raise KeyError(path)
        r.raise_for_status()
        return r.text, r.headers.get("Location", "")

    async def whip_publish(self, path: str, offer_sdp: str) -> tuple[str, str]:
        return await self._signal(path, "whip", offer_sdp)

    async def whep_play(self, path: str, offer_sdp: str) -> tuple[str, str]:
        return await self._signal(path, "whep", offer_sdp)

    async def _delete(self, location: str) -> None:
        if not location:
            return
        url = location if location.startswith("http") else f"{self.webrtc}{location}"
        try:
            await self._client.delete(url)
        except httpx.HTTPError:
            pass

    async def whip_delete(self, session_id: str) -> None:
        await self._delete(session_id)

    async def whep_delete(self, session_id: str) -> None:
        await self._delete(session_id)

    def streams(self) -> dict[str, StreamInfo]:
        return self._cache

    _cache: dict[str, StreamInfo] = {}

    async def refresh(self) -> None:
        try:
            r = await self._client.get(f"{self.api}/v3/paths/list")
            items = r.json().get("items", [])
        except Exception:
            return
        self._cache = {
            it["name"]: StreamInfo(
                path=it["name"], live=it.get("ready", False),
                viewers=len(it.get("readers", [])),
                recording=it["name"] in self._recording,
                tracks=it.get("tracks", []),
            )
            for it in items
        }

    async def frames(self, path: str, fps: float = 5.0):
        """Decode the stream over local RTSP with OpenCV in a worker thread."""
        import cv2

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        stop = False

        def worker():
            cap = cv2.VideoCapture(f"rtsp://127.0.0.1:8554/{path}", cv2.CAP_FFMPEG)
            last = 0.0
            try:
                while not stop and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    now = time.time()
                    if now - last < 1.0 / fps:
                        continue
                    last = now
                    if q.full():
                        continue
                    loop.call_soon_threadsafe(q.put_nowait, (now, frame))
            finally:
                cap.release()
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, worker)
        try:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            stop = True
            await fut

    async def add_source(self, path: str, url: str) -> None:
        body = {"source": url, "sourceOnDemand": False, "rtspTransport": "tcp"}
        r = await self._client.post(f"{self.api}/v3/config/paths/add/{path}", json=body)
        if r.status_code == 400 and "already" in r.text:
            r = await self._client.patch(f"{self.api}/v3/config/paths/patch/{path}", json=body)
        r.raise_for_status()

    async def remove_source(self, path: str) -> None:
        await self._client.delete(f"{self.api}/v3/config/paths/delete/{path}")

    async def _set_record(self, path: str, on: bool) -> None:
        body = {"record": on}
        r = await self._client.patch(f"{self.api}/v3/config/paths/patch/{path}", json=body)
        if r.status_code == 404:  # path only exists via all_others: create an explicit entry
            r = await self._client.post(f"{self.api}/v3/config/paths/add/{path}", json=body)
        r.raise_for_status()

    async def start_recording(self, path: str, file: Path) -> None:
        await self._set_record(path, True)
        self._recording.add(path)

    async def stop_recording(self, path: str) -> None:
        await self._set_record(path, False)
        self._recording.discard(path)

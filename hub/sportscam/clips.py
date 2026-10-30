"""Highlight clips: cut ±N seconds around each tagged event from the recording of the best camera.

The hub keeps an index of recordings (path, file, wall-clock start) in recordings/<code>/index.json
so clips can be built during the game, right after it, or days later. Cutting uses ffmpeg with a
fast re-encode so clip boundaries are exact (stream copy would snap to keyframes).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger("sportscam.clips")

PRE_S, POST_S = 8.0, 6.0
KIND_SCORE = {"goal": 100, "penalty": 60, "hit": 70, "save": 50, "shot": 40, "highlight": 30, "score": 0, "period": 0}


class RecordingIndex:
    def __init__(self, recordings_dir: Path):
        self.dir = recordings_dir

    def _file(self, code: str) -> Path:
        return self.dir / code / "index.json"

    def load(self, code: str) -> list[dict]:
        f = self._file(code)
        return json.loads(f.read_text()) if f.exists() else []

    def add(self, code: str, cam_id: str, file: Path, start_wall: float) -> None:
        items = self.load(code)
        items.append({"cam_id": cam_id, "file": file.name, "start": start_wall})
        self._file(code).parent.mkdir(parents=True, exist_ok=True)
        self._file(code).write_text(json.dumps(items))

    def find(self, code: str, cam_id: str, wall: float) -> dict | None:
        """Recording of cam_id that covers wall time (latest one that started before it)."""
        cands = [r for r in self.load(code) if r["cam_id"] == cam_id and r["start"] <= wall]
        return max(cands, key=lambda r: r["start"]) if cands else None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def cut_clip(src: Path, start_s: float, duration_s: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start_s):.2f}", "-i", str(src),
           "-t", f"{duration_s:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-movflags", "+faststart", str(out)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        log.warning("ffmpeg failed for %s: %s", out.name, err.decode()[-300:])
        return False
    return out.exists() and out.stat().st_size > 1000


class ClipBuilder:
    def __init__(self, recordings_dir: Path, index: RecordingIndex):
        self.dir = recordings_dir
        self.index = index
        self._jobs: dict[str, asyncio.Task] = {}

    def clips_dir(self, code: str) -> Path:
        return self.dir / code / "clips"

    def manifest(self, code: str) -> list[dict]:
        f = self.clips_dir(code) / "clips.json"
        return json.loads(f.read_text()) if f.exists() else []

    def building(self, code: str) -> bool:
        j = self._jobs.get(code)
        return bool(j and not j.done())

    def build(self, code: str, events: list[dict], pick_camera, cameras: dict[str, str]) -> None:
        """Start (or restart) a background build. pick_camera(event) -> cam_id | None."""
        if self.building(code):
            return
        self._jobs[code] = asyncio.create_task(self._build(code, events, pick_camera, cameras))

    async def _build(self, code: str, events: list[dict], pick_camera, cameras: dict[str, str]) -> None:
        out_dir = self.clips_dir(code)
        done = {c["event_id"] for c in self.manifest(code)}
        clips = self.manifest(code)
        for ev in events:
            if ev["kind"] not in KIND_SCORE or KIND_SCORE[ev["kind"]] == 0 or ev["id"] in done:
                continue
            cam_id = pick_camera(ev)
            rec = self.index.find(code, cam_id, ev["wall"]) if cam_id else None
            if not rec:
                # fall back to any camera that has a recording covering the moment
                for cid in cameras:
                    rec = self.index.find(code, cid, ev["wall"])
                    if rec:
                        cam_id = cid
                        break
            if not rec:
                continue
            src = self.dir / code / rec["file"]
            offset = ev["wall"] - rec["start"] - PRE_S
            out = out_dir / f"{ev['id']:04d}_{ev['kind']}_{cam_id}.mp4"
            ok = await cut_clip(src, offset, PRE_S + POST_S, out)
            if ok:
                clips.append({"event_id": ev["id"], "kind": ev["kind"], "team": ev.get("team"), "t_ms": ev["t_ms"],
                              "camera_id": cam_id, "camera": cameras.get(cam_id, cam_id), "file": out.name,
                              "score": KIND_SCORE[ev["kind"]], "built_at": time.time()})
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "clips.json").write_text(json.dumps(clips))
        log.info("clips for %s: %d", code, len(clips))

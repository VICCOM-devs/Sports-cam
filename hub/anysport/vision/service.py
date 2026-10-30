"""Live vision service: one worker per live camera, one fusion per match.

Per camera (at `fps`): decode frame → detect people → track → jersey feature → if calibrated,
project skate point to rink metres. Per match (every tick): fuse all cameras into players and
broadcast `positions` over the match WebSocket. On match end, write tracking.json with paths
and stats next to the recordings.

Detection runs in a thread pool so the event loop (signalling, WebSockets) never waits on it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detector import PersonDetector
from .fusion import Fusion, Observation
from .rink import TEMPLATES, Calibration
from .teams import TeamClassifier, torso_feature
from .tracker import Tracker

log = logging.getLogger("anysport.vision")


@dataclass
class CameraWorker:
    cam_id: str
    path: str
    task: asyncio.Task
    tracker: Tracker
    calibration: Calibration | None = None
    last_obs: list[Observation] = field(default_factory=list)
    last_frame_t: float = 0.0
    fps_measured: float = 0.0
    detections: int = 0
    frame_size: tuple[int, int] = (0, 0)
    crop: list[float] | None = None     # normalized [x, y, w, h] around the play, smoothed
    overlay: list[dict] = field(default_factory=list)   # per-track drawing data in image pixels


@dataclass
class MatchVision:
    match_id: str
    code: str
    fusion: Fusion
    teams: TeamClassifier = field(default_factory=TeamClassifier)
    workers: dict[str, CameraWorker] = field(default_factory=dict)
    feats: list[np.ndarray] = field(default_factory=list)
    samples: dict[str, list[bytes]] = field(default_factory=lambda: {"A": [], "B": []})
    tick_task: asyncio.Task | None = None
    started: float = field(default_factory=time.time)
    director_cam: str | None = None
    director_since: float = 0.0
    director_log: list[tuple[float, str]] = field(default_factory=list)   # (wall, cam_id)


def _overlay_entries(mv: MatchVision, w: CameraWorker, tracks) -> list[dict]:
    """What the viewer needs to draw broadcast-style graphics on this camera's picture:
    box, foot point, team, trail (last foot points), and the fused player's id/label/speed."""
    by_track = {}
    for p in mv.fusion.players.values():
        tid = p.sources.get(w.cam_id)
        if tid is not None:
            by_track[tid] = p
    out = []
    for tr in tracks:
        x1, y1, x2, y2 = (round(float(v)) for v in tr.box)
        fx, fy = tr.foot
        p = by_track.get(tr.id)
        if p is not None and p.opted_out:
            continue
        e = {"tid": tr.id, "box": [x1, y1, x2, y2], "foot": [round(fx), round(fy)], "team": tr.team,
             "trail": [[round(a), round(b)] for a, b in tr.history[-12:]]}
        if p is not None:
            e.update({"pid": p.id, "label": p.label, "speed": round(p.speed_kmh), "dist": round(p.distance_m)})
        out.append(e)
    return out


def _follow_crop(prev: list[float] | None, tracks, W: int, H: int, min_w: float = 0.45, margin: float = 0.12,
                 alpha: float = 0.15) -> list[float] | None:
    """Virtual follow-cam: a 16:9 window around the tracked players, never smaller than min_w of
    the frame, eased towards the target so the zoom glides instead of jumping."""
    if not tracks or not W or not H:
        return None if prev is None else prev
    xs1 = min(t.box[0] for t in tracks) / W; ys1 = min(t.box[1] for t in tracks) / H
    xs2 = max(t.box[2] for t in tracks) / W; ys2 = max(t.box[3] for t in tracks) / H
    cx, cy = (xs1 + xs2) / 2, (ys1 + ys2) / 2
    w = max(min_w, (xs2 - xs1) + 2 * margin)
    h = max(min_w, (ys2 - ys1) + 2 * margin)
    w = max(w, h)          # keep the frame's aspect (both normalized → same ratio as the frame)
    h = w
    w = min(w, 1.0); h = min(h, 1.0)
    x = min(max(cx - w / 2, 0.0), 1.0 - w); y = min(max(cy - h / 2, 0.0), 1.0 - h)
    target = [x, y, w, h]
    if prev is None:
        return target
    return [p + alpha * (t - p) for p, t in zip(prev, target)]


class VisionService:
    def __init__(self, media, bus, models_dir: Path, recordings_dir: Path, model: str = "nano",
                 fps: float = 5.0, threads: int = 2):
        self.media = media
        self.bus = bus
        self.models_dir = models_dir
        self.recordings_dir = recordings_dir
        self.model_size = model
        self.fps = fps
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="vision")
        self.detector: PersonDetector | None = None
        self.matches: dict[str, MatchVision] = {}
        self.enabled = True

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.detector = await loop.run_in_executor(self.pool, PersonDetector.load, self.models_dir, self.model_size)
            log.info("detector ready (%s)", self.model_size)
        except Exception as e:
            self.enabled = False
            log.warning("vision disabled: %s", e)

    async def stop(self) -> None:
        for mv in list(self.matches.values()):
            await self.stop_match(mv.match_id)
        self.pool.shutdown(wait=False)

    # -- lifecycle -----------------------------------------------------------
    def ensure_match(self, m: dict) -> MatchVision:
        mv = self.matches.get(m["id"])
        if not mv:
            mv = MatchVision(m["id"], m["code"], Fusion(rink=TEMPLATES["iihf"]))
            if m.get("team_map"):
                mv.teams.load(json.loads(m["team_map"]))
            mv.tick_task = asyncio.create_task(self._tick(mv))
            self.matches[m["id"]] = mv
        return mv

    def ensure_camera(self, m: dict, cam: dict, path: str) -> None:
        if not self.enabled:
            return
        mv = self.ensure_match(m)
        w = mv.workers.get(cam["id"])
        if w and not w.task.done():
            return
        cal = Calibration.from_dict(json.loads(cam["calibration"])) if cam.get("calibration") else None
        w = CameraWorker(cam["id"], path, None, Tracker(dt=1.0 / self.fps), cal)  # type: ignore[arg-type]
        w.task = asyncio.create_task(self._camera_loop(mv, w))
        mv.workers[cam["id"]] = w

    def set_calibration(self, match_id: str, cam_id: str, cal: Calibration | None) -> None:
        mv = self.matches.get(match_id)
        if mv and cam_id in mv.workers:
            mv.workers[cam_id].calibration = cal

    def set_team_map(self, match_id: str, mapping: dict[str, str]) -> None:
        mv = self.matches.get(match_id)
        if mv:
            mv.teams.mapping = mapping

    async def stop_match(self, match_id: str) -> dict | None:
        mv = self.matches.pop(match_id, None)
        if not mv:
            return None
        for w in mv.workers.values():
            w.task.cancel()
        if mv.tick_task:
            mv.tick_task.cancel()
        summary = {"code": mv.code, "players": mv.fusion.stats(), "teams": mv.teams.to_dict(),
                   "director_log": mv.director_log,
                   "cameras": {cid: {"detections": w.detections, "calibrated": w.calibration is not None}
                               for cid, w in mv.workers.items()}}
        out = self.recordings_dir / mv.code / "tracking.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary))
        return summary

    # -- status --------------------------------------------------------------
    def status(self, match_id: str) -> dict:
        mv = self.matches.get(match_id)
        if not mv:
            return {"enabled": self.enabled, "cameras": {}, "players": 0, "teams_ready": False}
        return {
            "enabled": self.enabled,
            "cameras": {cid: {"fps": round(w.fps_measured, 1), "calibrated": w.calibration is not None,
                              "tracks": len([t for t in w.tracker.tracks if t.confirmed]), "detections": w.detections}
                        for cid, w in mv.workers.items()},
            "players": len(mv.fusion.players),
            "teams_ready": mv.teams.centers is not None,
            "team_map": mv.teams.mapping,
        }

    def director(self, match_id: str) -> str | None:
        mv = self.matches.get(match_id)
        return mv.director_cam if mv else None

    def director_at(self, match_id: str, wall: float) -> str | None:
        """Camera the director had chosen at a given wall time (for clips)."""
        mv = self.matches.get(match_id)
        if not mv:
            return None
        pick = None
        for t, cam in mv.director_log:
            if t <= wall:
                pick = cam
            else:
                break
        return pick

    def _pick_director(self, mv: MatchVision, now: float, min_shot_s: float = 4.0) -> None:
        """Best angle = camera that sees the most confirmed tracks right now, with hysteresis so
        the feed doesn't flicker. Calibrated cameras get a bonus when the action centroid is in
        their view (the fused players tell us where the play is)."""
        scores: dict[str, float] = {}
        players = [p for p in mv.fusion.players.values() if p.misses == 0]
        cx = sum(p.x for p in players) / len(players) if players else None
        cy = sum(p.y for p in players) / len(players) if players else None
        for cid, w in mv.workers.items():
            if now - w.last_frame_t > 2:
                continue
            n = len([t for t in w.tracker.tracks if t.confirmed and t.misses == 0])
            score = float(n)
            if w.calibration and cx is not None:
                px, py = w.calibration.rink_to_image(cx, cy)
                if 0 <= px <= 1920 and 0 <= py <= 1080:
                    score += 2.0
            scores[cid] = score
        if not scores:
            return
        best = max(scores, key=scores.get)
        cur = mv.director_cam
        if cur is None or cur not in scores:
            switch = True
        else:
            switch = scores[best] > scores[cur] * 1.3 + 0.5 and now - mv.director_since >= min_shot_s
        if switch and best != cur:
            mv.director_cam, mv.director_since = best, now
            mv.director_log.append((now, best))
            if len(mv.director_log) > 5000:
                del mv.director_log[:-5000]

    def label_player(self, match_id: str, player_id: int, label: str | None, opted_out: bool) -> bool:
        mv = self.matches.get(match_id)
        p = mv.fusion.players.get(player_id) if mv else None
        if not p:
            return False
        p.label, p.opted_out = label, opted_out
        return True

    def positions(self, match_id: str) -> list[dict]:
        mv = self.matches.get(match_id)
        return mv.fusion.snapshot() if mv else []

    def stats(self, match_id: str) -> list[dict]:
        mv = self.matches.get(match_id)
        return mv.fusion.stats() if mv else []

    def team_samples(self, match_id: str) -> dict[str, list[bytes]]:
        mv = self.matches.get(match_id)
        return mv.samples if mv else {"A": [], "B": []}

    # -- loops ---------------------------------------------------------------
    async def _camera_loop(self, mv: MatchVision, w: CameraWorker) -> None:
        loop = asyncio.get_running_loop()
        import cv2

        n, t0 = 0, time.time()
        try:
            async for t, frame in self.media.frames(w.path, fps=self.fps):
                dets = await loop.run_in_executor(self.pool, self.detector, frame)
                w.detections += len(dets)
                tracks = w.tracker.step(dets)
                obs = []
                for tr in tracks:
                    feat = torso_feature(frame, tr.box)
                    if feat is not None:
                        tr.color = feat
                        if mv.teams.centers is None and len(mv.feats) < 400:
                            mv.feats.append(feat)
                        key = mv.teams.cluster(feat)
                        tr.team = mv.teams.mapping.get(key, key) if key else None
                        # keep a few jersey crops per cluster for the admin's team confirmation
                        if key in ("A", "B") and len(mv.samples[key]) < 6 and n % 10 == 0:
                            x1, y1, x2, y2 = (int(v) for v in tr.box)
                            crop = frame[max(0, y1):y2, max(0, x1):x2]
                            if crop.size:
                                ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (64, 128)))
                                if ok:
                                    mv.samples[key].append(buf.tobytes())
                    if w.calibration:
                        fx, fy = tr.foot
                        x, y = w.calibration.image_to_rink(fx, fy)
                        obs.append(Observation(w.cam_id, tr.id, x, y, tr.team, t))
                w.last_obs = obs
                w.last_frame_t = t
                w.frame_size = (frame.shape[1], frame.shape[0])
                w.overlay = _overlay_entries(mv, w, tracks)
                w.crop = _follow_crop(w.crop, tracks, frame.shape[1], frame.shape[0])
                n += 1
                if n % 25 == 0:
                    w.fps_measured = 25 / (time.time() - t0); t0 = time.time()
                if mv.teams.centers is None and len(mv.feats) >= 60 and n % 25 == 0:
                    if mv.teams.fit(mv.feats):
                        log.info("match %s: team colours clustered", mv.code)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("camera loop %s failed", w.cam_id)

    async def _tick(self, mv: MatchVision) -> None:
        period = 1.0 / self.fps
        try:
            while True:
                await asyncio.sleep(period)
                now = time.time()
                obs = [o for w in mv.workers.values() if now - w.last_frame_t < 2 for o in w.last_obs]
                mv.fusion.step(obs, now)
                self._pick_director(mv, now)
                if self.bus.count(mv.match_id):
                    await self.bus.publish(mv.match_id, "positions",
                                           {"t": now, "players": mv.fusion.snapshot(), "director": mv.director_cam,
                                            "crops": {cid: [round(v, 3) for v in w.crop] for cid, w in mv.workers.items()
                                                      if w.crop and now - w.last_frame_t < 2},
                                            "overlay": {cid: {"t": w.last_frame_t, "size": w.frame_size, "tracks": w.overlay}
                                                        for cid, w in mv.workers.items() if now - w.last_frame_t < 2}})
        except asyncio.CancelledError:
            pass

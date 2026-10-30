"""The hub: one FastAPI app serving the API, WebSocket events, WHIP/WHEP signalling and the web apps."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import qrcode
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import codes, version
from .config import Config
from .db import Database
from .events import EventBus
from .media.base import MediaEngine
from .updates import UpdateChecker
from .clips import ClipBuilder, RecordingIndex
from .vision.rink import TEMPLATES, Calibration, calibrate
from .vision.service import VisionService

log = logging.getLogger("sportscam")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

Role = Literal["admin", "scorekeeper", "camera", "viewer"]
TAG_KINDS = {"goal", "shot", "save", "hit", "penalty", "highlight"}
ROLE_RANK = {"viewer": 0, "camera": 1, "scorekeeper": 2, "admin": 3}


# --------------------------------------------------------------------------- app state
class Hub:
    def __init__(self, cfg: Config, media: MediaEngine):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = Database(cfg.db_path)
        self.bus = EventBus()
        self.media = media
        self.updates = UpdateChecker(cfg.update_url)
        self.vision = VisionService(media, self.bus, cfg.models_dir, cfg.recordings_dir,
                                    model=cfg.vision_model, fps=cfg.vision_fps)
        if not cfg.vision:
            self.vision.enabled = False
        self.rec_index = RecordingIndex(cfg.recordings_dir)
        self.clips = ClipBuilder(cfg.recordings_dir, self.rec_index)

    # -- helpers
    def public_url(self, secure: bool = False) -> str:
        if secure:
            return f"https://{self.cfg.public_host}:{self.cfg.https_port}"
        return f"http://{self.cfg.public_host}:{self.cfg.http_port}"

    def clock_ms(self, m: dict) -> int:
        base = int(m["clock_base_ms"] or 0)
        if m["clock_running"] and m["clock_started_at"]:
            base += int((time.time() - m["clock_started_at"]) * 1000)
        return base

    def match_state(self, m: dict) -> dict:
        streams = self.media.streams()
        cams = []
        for c in self.db.cameras(m["id"]):
            s = streams.get(stream_path(m, c["id"]))
            c = dict(c)
            c["live"] = bool(s and s.live)
            c["viewers"] = s.viewers if s else 0
            c["recording"] = bool(s and s.recording)
            c["calibrated"] = bool(c.pop("calibration", None))
            cams.append(c)
        return {
            "id": m["id"], "code": m["code"], "sport": m["sport"],
            "home": m["home"], "away": m["away"],
            "home_color": m["home_color"], "away_color": m["away_color"],
            "status": m["status"], "period": m["period"],
            "home_score": m["home_score"], "away_score": m["away_score"],
            "clock_running": bool(m["clock_running"]), "clock_ms": self.clock_ms(m),
            "cameras": cams,
            "viewers": self.bus.count(m["id"]),
            "server_time": time.time(),
            "vision": self.vision.status(m["id"]),
        }

    async def broadcast_state(self, match_id: str) -> None:
        m = self.db.match(match_id)
        if m:
            await self.bus.publish(match_id, "state", self.match_state(m))


def stream_path(m: dict, cam_id: str) -> str:
    return f"m/{m['code']}/{cam_id}"


def build_media(cfg: Config) -> MediaEngine:
    from .media import mediamtx, pyrtc

    engine = cfg.media_engine
    if engine == "auto":
        engine = "mediamtx" if mediamtx.binary_path(cfg.bin_dir).exists() else "pyrtc"
    if engine == "mediamtx":
        return mediamtx.MediaMtxEngine(cfg.bin_dir, cfg.recordings_dir, cfg.mediamtx_api, cfg.mediamtx_webrtc)
    return pyrtc.PyRtcEngine()


def create_app(cfg: Config | None = None, media: MediaEngine | None = None) -> FastAPI:
    cfg = cfg or Config()
    hub = Hub(cfg, media or build_media(cfg))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await hub.media.start()
        if hub.cfg.vision:
            await hub.vision.start()
        hub.updates.start()
        poll = asyncio.create_task(_poll_cameras(hub))
        log.info("%s %s (build %s) on %s and %s, media=%s", version.DISPLAY_NAME, version.SEMVER,
                 version.BUILD_NUMBER, hub.public_url(), hub.public_url(True), hub.media.name)
        try:
            yield
        finally:
            poll.cancel()
            await hub.updates.stop()
            await hub.vision.stop()
            await hub.media.stop()

    app = FastAPI(title=version.DISPLAY_NAME, version=version.SEMVER, lifespan=lifespan)
    app.state.hub = hub
    register_routes(app, hub)
    return app


async def _poll_cameras(hub: Hub) -> None:
    """Every 2 s: refresh stream info, mark silent cameras offline, push state to viewers."""
    while True:
        try:
            if hasattr(hub.media, "refresh"):
                await hub.media.refresh()  # type: ignore[attr-defined]
            streams = hub.media.streams()
            now = time.time()
            for m in hub.db.recent_matches(5):
                if m["status"] == "ended":
                    continue
                changed = False
                for c in hub.db.cameras(m["id"]):
                    s = streams.get(stream_path(m, c["id"]))
                    status = "live" if s and s.live else ("offline" if now - c["last_seen"] > 15 else "joined")
                    if status != c["status"]:
                        hub.db.update_camera(c["id"], status=status)
                        changed = True
                    if s and s.live and m["status"] == "live" and not s.recording and hub.cfg.record:
                        await _record(hub, m, c["id"])
                    if s and s.live and hub.vision.enabled:
                        hub.vision.ensure_camera(m, c, stream_path(m, c["id"]))
                if changed or hub.bus.count(m["id"]):
                    await hub.broadcast_state(m["id"])
        except Exception:
            log.exception("camera poll failed")
        await asyncio.sleep(2)


async def _record(hub: Hub, m: dict, cam_id: str) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Matroska: written incrementally, so the file stays readable even if the stop is ungraceful
    file = hub.cfg.recordings_dir / m["code"] / f"{cam_id}_{stamp}.mkv"
    try:
        await hub.media.start_recording(stream_path(m, cam_id), file)
        hub.rec_index.add(m["code"], cam_id, file, time.time())
    except Exception as e:
        log.warning("could not start recording %s: %s", cam_id, e)


def _build_clips(hub: Hub, m: dict) -> None:
    cams = {c["id"]: c["label"] for c in hub.db.cameras(m["id"])}

    def pick(ev: dict) -> str | None:
        cam = (ev.get("payload") or {}).get("camera")
        if cam and cam in cams:
            return cam
        return hub.vision.director_at(m["id"], ev["wall"]) or (next(iter(cams)) if cams else None)

    hub.clips.build(m["code"], hub.db.events(m["id"]), pick, cams)


# --------------------------------------------------------------------------- models
class MatchCreate(BaseModel):
    sport: str = "icehockey"
    home: str = Field(min_length=1, max_length=40)
    away: str = Field(min_length=1, max_length=40)
    home_color: str = "#1E6FE8"
    away_color: str = "#E53935"


class JoinBody(BaseModel):
    code: str
    role: Literal["scorekeeper", "camera", "viewer"] = "viewer"
    label: str | None = Field(default=None, max_length=40)


class CameraRegister(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    position: str | None = Field(default=None, max_length=40)
    kind: Literal["phone", "rink", "capture"] = "phone"


class CameraStatus(BaseModel):
    battery: float | None = None
    signal: str | None = None


class TagBody(BaseModel):
    kind: str
    team: Literal["home", "away"] | None = None
    payload: dict = Field(default_factory=dict)


class ScoreBody(BaseModel):
    home_score: int | None = None
    away_score: int | None = None
    period: int | None = None


class ClockBody(BaseModel):
    running: bool
    set_ms: int | None = None


class LandmarkPair(BaseModel):
    id: str
    px: float
    py: float


class CalibrationBody(BaseModel):
    template: str = "iihf"
    pairs: list[LandmarkPair] = Field(min_length=4, max_length=40)


class TeamMapBody(BaseModel):
    A: Literal["home", "away"]
    B: Literal["home", "away"]


class RosterBody(BaseModel):
    team: Literal["home", "away"]
    number: str | None = Field(default=None, max_length=4)
    name: str = Field(min_length=1, max_length=40)
    opted_out: bool = False


class RosterPatch(BaseModel):
    team: Literal["home", "away"] | None = None
    number: str | None = Field(default=None, max_length=4)
    name: str | None = Field(default=None, max_length=40)
    opted_out: bool | None = None
    player_id: int | None = None     # link to a tracked player; 0 clears


# --------------------------------------------------------------------------- routes
def register_routes(app: FastAPI, hub: Hub) -> None:
    def is_local(request: Request) -> bool:
        host = request.client.host if request.client else ""
        return host in ("127.0.0.1", "::1", "localhost", "testclient")

    def auth(min_role: Role = "viewer"):
        def dep(request: Request, x_token: str | None = Header(default=None), t: str | None = Query(default=None)):
            tok = x_token or t
            row = hub.db.token(tok) if tok else None
            if not row:
                raise HTTPException(401, "invalid token")
            if ROLE_RANK[row["role"]] < ROLE_RANK[min_role]:
                raise HTTPException(403, "role not allowed")
            m = hub.db.match(row["match_id"])
            if not m:
                raise HTTPException(404, "match gone")
            return row, m
        return dep

    # -- version / updates
    @app.get("/api/version")
    async def api_version():
        return {
            "product": version.PRODUCT, "display_name": version.DISPLAY_NAME,
            "semver": version.SEMVER, "build": version.BUILD_NUMBER,
            "channel": version.CHANNEL, "media": hub.media.name,
            "secure_url": hub.public_url(True),
            "update": hub.updates.state,
        }

    @app.get("/api/hub")
    async def api_hub(request: Request):
        if not is_local(request):
            raise HTTPException(403, "hub console is local only")
        return {
            "url": hub.public_url(), "secure_url": hub.public_url(True),
            "host": hub.cfg.public_host, "media": hub.media.name,
            "recordings_dir": str(hub.cfg.recordings_dir),
            "matches": [hub.match_state(m) for m in hub.db.recent_matches(10)],
        }

    # -- matches (local hub console only)
    @app.post("/api/matches", status_code=201)
    async def create_match(body: MatchCreate, request: Request):
        if not is_local(request):
            raise HTTPException(403, "matches are created on the hub")
        code = codes.match_code()
        while hub.db.match_by_code(code):
            code = codes.match_code()
        mid = uuid.uuid4().hex[:12]
        hub.db.insert_match({
            "id": mid, "code": code, "sport": body.sport, "home": body.home, "away": body.away,
            "home_color": body.home_color, "away_color": body.away_color,
            "status": "lobby", "created_at": time.time(),
        })
        admin = codes.token()
        hub.db.insert_token(admin, mid, "admin", "hub")
        m = hub.db.match(mid)
        return {"match": hub.match_state(m), "admin_token": admin,
                "join_url": f"{hub.public_url()}/?c={code}",
                "camera_url": f"{hub.public_url(True)}/camera.html?c={code}"}

    @app.post("/api/join")
    async def join(body: JoinBody):
        m = hub.db.match_by_code(codes.normalize_code(body.code))
        if not m:
            raise HTTPException(404, "no match with that code")
        if m["status"] == "ended" and body.role != "viewer":
            raise HTTPException(409, "match has ended")
        tok = codes.token()
        hub.db.insert_token(tok, m["id"], body.role, body.label)
        return {"token": tok, "role": body.role, "match": hub.match_state(m),
                "whep_base": "/whep", "ws": "/ws"}

    @app.get("/api/match")
    async def get_match(dep=Depends(auth("viewer"))):
        row, m = dep
        st = hub.match_state(m)
        st["role"] = row["role"]
        st["events"] = hub.db.events(m["id"])
        return st

    @app.post("/api/match/start")
    async def start_match(dep=Depends(auth("scorekeeper"))):
        _, m = dep
        if m["status"] == "lobby":
            hub.db.update_match(m["id"], status="live", started_at=time.time())
            for c in hub.db.cameras(m["id"]):
                if c["status"] == "live" and hub.cfg.record:
                    await _record(hub, m, c["id"])
        await hub.broadcast_state(m["id"])
        return hub.match_state(hub.db.match(m["id"]))

    @app.post("/api/match/end")
    async def end_match(dep=Depends(auth("admin"))):
        _, m = dep
        m2 = hub.db.match(m["id"])
        hub.db.update_match(m["id"], status="ended", ended_at=time.time(),
                            clock_running=0, clock_base_ms=hub.clock_ms(m2))
        for c in hub.db.cameras(m["id"]):
            await hub.media.stop_recording(stream_path(m, c["id"]))
        _build_clips(hub, m)
        await hub.vision.stop_match(m["id"])
        await hub.broadcast_state(m["id"])
        return hub.match_state(hub.db.match(m["id"]))

    @app.post("/api/match/clock")
    async def clock(body: ClockBody, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        base = body.set_ms if body.set_ms is not None else hub.clock_ms(m)
        hub.db.update_match(m["id"], clock_running=int(body.running),
                            clock_base_ms=base, clock_started_at=time.time() if body.running else None)
        await hub.broadcast_state(m["id"])
        return hub.match_state(hub.db.match(m["id"]))

    @app.post("/api/match/score")
    async def score(body: ScoreBody, dep=Depends(auth("scorekeeper"))):
        row, m = dep
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if not fields:
            raise HTTPException(400, "nothing to set")
        hub.db.update_match(m["id"], **fields)
        m2 = hub.db.match(m["id"])
        kind = "period" if "period" in fields and len(fields) == 1 else "score"
        ev = hub.db.insert_event(m["id"], hub.clock_ms(m2), kind, None,
                                 {"home": m2["home_score"], "away": m2["away_score"], "period": m2["period"]},
                                 row["token"])
        await hub.bus.publish(m["id"], "event", ev)
        await hub.broadcast_state(m["id"])
        return hub.match_state(m2)

    # -- tags
    @app.post("/api/events", status_code=201)
    async def add_event(body: TagBody, dep=Depends(auth("viewer"))):
        row, m = dep
        if body.kind not in TAG_KINDS:
            raise HTTPException(400, f"kind must be one of {sorted(TAG_KINDS)}")
        if body.kind != "highlight" and ROLE_RANK[row["role"]] < ROLE_RANK["scorekeeper"]:
            raise HTTPException(403, "viewers can only tag highlights")
        ev = hub.db.insert_event(m["id"], hub.clock_ms(m), body.kind, body.team, body.payload, row["token"])
        await hub.bus.publish(m["id"], "event", ev)
        return ev

    @app.get("/api/events")
    async def list_events(dep=Depends(auth("viewer"))):
        _, m = dep
        return hub.db.events(m["id"])

    # -- cameras
    @app.post("/api/cameras", status_code=201)
    async def register_camera(body: CameraRegister, dep=Depends(auth("camera"))):
        row, m = dep
        cam_id = "c" + uuid.uuid4().hex[:8]
        now = time.time()
        hub.db.upsert_camera({
            "id": cam_id, "match_id": m["id"], "label": body.label, "position": body.position,
            "kind": body.kind, "status": "joined", "joined_at": now, "last_seen": now,
        })
        await hub.broadcast_state(m["id"])
        return {"camera_id": cam_id, "whip": f"/whip/{cam_id}", "path": stream_path(m, cam_id)}

    @app.post("/api/cameras/{cam_id}/status")
    async def camera_status(cam_id: str, body: CameraStatus, dep=Depends(auth("camera"))):
        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        hub.db.update_camera(cam_id, last_seen=time.time(), battery=body.battery, signal=body.signal)
        return {"ok": True}

    # -- rink, calibration, tracking
    @app.get("/api/rink")
    async def rink(template: str = "iihf"):
        if template not in TEMPLATES:
            raise HTTPException(404, "unknown rink template")
        return TEMPLATES[template].to_dict()

    @app.get("/api/cameras/{cam_id}/snapshot.jpg")
    async def snapshot(cam_id: str, dep=Depends(auth("scorekeeper"))):
        import cv2

        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        frame = await hub.media.snapshot(stream_path(m, cam_id))
        if frame is None:
            raise HTTPException(404, "camera is not live")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{cam_id}/calibration")
    async def get_calibration(cam_id: str, dep=Depends(auth("viewer"))):
        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        return json.loads(c["calibration"]) if c.get("calibration") else None

    @app.post("/api/cameras/{cam_id}/calibration")
    async def set_calibration(cam_id: str, body: CalibrationBody, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        if body.template not in TEMPLATES:
            raise HTTPException(400, "unknown rink template")
        try:
            cal = calibrate(TEMPLATES[body.template], [p.model_dump() for p in body.pairs])
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.db.update_camera(cam_id, calibration=json.dumps(cal.to_dict()))
        hub.vision.set_calibration(m["id"], cam_id, cal)
        await hub.broadcast_state(m["id"])
        # projected rink lines in image pixels, so the client can draw the overlay
        tpl = TEMPLATES[body.template]
        overlay = {"lines": [[cal.rink_to_image(*a), cal.rink_to_image(*b)] for a, b in tpl.lines()],
                   "outline": [cal.rink_to_image(x, y) for x, y in tpl.outline()]}
        return {"rms_error_m": cal.rms_error_m, "overlay": overlay}

    @app.delete("/api/cameras/{cam_id}/calibration")
    async def clear_calibration(cam_id: str, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        hub.db.update_camera(cam_id, calibration=None)
        hub.vision.set_calibration(m["id"], cam_id, None)
        return Response(status_code=204)

    @app.get("/api/positions")
    async def positions(dep=Depends(auth("viewer"))):
        _, m = dep
        return {"t": time.time(), "players": hub.vision.positions(m["id"])}

    @app.get("/api/stats")
    async def stats(dep=Depends(auth("viewer"))):
        _, m = dep
        live = hub.vision.stats(m["id"])
        if live:
            return {"source": "live", "players": live}
        f = hub.cfg.recordings_dir / m["code"] / "tracking.json"
        if f.exists():
            return {"source": "file", **json.loads(f.read_text())}
        return {"source": "none", "players": []}

    @app.get("/api/teams/samples")
    async def team_samples(dep=Depends(auth("scorekeeper"))):
        import base64

        _, m = dep
        s = hub.vision.team_samples(m["id"])
        return {k: [base64.b64encode(b).decode() for b in v] for k, v in s.items()}

    @app.post("/api/teams/map")
    async def team_map(body: TeamMapBody, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        mapping = {"A": body.A, "B": body.B}
        hub.db.update_match(m["id"], team_map=json.dumps({"mapping": mapping}))
        hub.vision.set_team_map(m["id"], mapping)
        await hub.broadcast_state(m["id"])
        return mapping

    # -- roster (names, numbers, opt-out). Opted-out players get no name, stats, path or clips as subject.
    def _apply_roster_link(m: dict, row: dict) -> None:
        if row.get("player_id"):
            label = None if row["opted_out"] else (f"#{row['number']} {row['name']}" if row.get("number") else row["name"])
            hub.vision.label_player(m["id"], int(row["player_id"]), label, bool(row["opted_out"]))

    @app.get("/api/roster")
    async def get_roster(dep=Depends(auth("viewer"))):
        row, m = dep
        items = hub.db.roster(m["id"])
        if ROLE_RANK[row["role"]] < ROLE_RANK["scorekeeper"]:
            items = [{k: v for k, v in r.items() if k != "opted_out"} for r in items if not r["opted_out"]]
        return items

    @app.post("/api/roster", status_code=201)
    async def add_roster(body: RosterBody, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        return hub.db.add_roster(m["id"], body.team, body.number, body.name, body.opted_out)

    @app.patch("/api/roster/{rid}")
    async def patch_roster(rid: int, body: RosterPatch, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if "player_id" in fields:
            if fields["player_id"] == 0:
                fields["player_id"] = None
            else:  # one tracked player can only be one person
                for other in hub.db.roster(m["id"]):
                    if other["player_id"] == fields["player_id"] and other["id"] != rid:
                        hub.db.update_roster(other["id"], player_id=None)
        if "opted_out" in fields:
            fields["opted_out"] = int(fields["opted_out"])
        row = hub.db.update_roster(rid, **fields)
        if not row or row["match_id"] != m["id"]:
            raise HTTPException(404, "no such roster entry")
        _apply_roster_link(m, row)
        await hub.broadcast_state(m["id"])
        return row

    @app.delete("/api/roster/{rid}")
    async def delete_roster(rid: int, dep=Depends(auth("scorekeeper"))):
        _, m = dep
        hub.db.delete_roster(rid)
        return Response(status_code=204)

    # -- clips
    @app.get("/api/clips")
    async def list_clips(dep=Depends(auth("viewer"))):
        _, m = dep
        items = hub.clips.manifest(m["code"])
        items = sorted(items, key=lambda c: (-c["score"], c["t_ms"]))
        return {"building": hub.clips.building(m["code"]), "clips": items}

    @app.post("/api/clips/build", status_code=202)
    async def build_clips(dep=Depends(auth("scorekeeper"))):
        _, m = dep
        _build_clips(hub, m)
        return {"building": True}

    @app.get("/clips/{file}")
    async def clip_file(file: str, dep=Depends(auth("viewer"))):
        _, m = dep
        path = hub.clips.clips_dir(m["code"]) / Path(file).name
        if not path.exists():
            raise HTTPException(404, "no such clip")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})

    # -- WHIP / WHEP signalling
    @app.post("/whip/{cam_id}")
    async def whip(cam_id: str, request: Request, dep=Depends(auth("camera"))):
        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        offer = (await request.body()).decode()
        answer, session = await hub.media.whip_publish(stream_path(m, cam_id), offer)
        hub.db.update_camera(cam_id, last_seen=time.time())
        return Response(answer, status_code=201, media_type="application/sdp",
                        headers={"Location": f"/whip/session/{session}"})

    @app.delete("/whip/session/{session}")
    async def whip_delete(session: str):
        await hub.media.whip_delete(session)
        return Response(status_code=204)

    @app.post("/whep/{cam_id}")
    async def whep(cam_id: str, request: Request, dep=Depends(auth("viewer"))):
        _, m = dep
        c = hub.db.camera(cam_id)
        if not c or c["match_id"] != m["id"]:
            raise HTTPException(404, "camera not in this match")
        offer = (await request.body()).decode()
        try:
            answer, session = await hub.media.whep_play(stream_path(m, cam_id), offer)
        except KeyError:
            raise HTTPException(404, "camera is not live")
        return Response(answer, status_code=201, media_type="application/sdp",
                        headers={"Location": f"/whep/session/{session}"})

    @app.delete("/whep/session/{session}")
    async def whep_delete(session: str):
        await hub.media.whep_delete(session)
        return Response(status_code=204)

    # -- live events
    @app.websocket("/ws")
    async def ws(websocket: WebSocket, t: str = Query(...)):
        row = hub.db.token(t)
        if not row:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        mid = row["match_id"]
        await hub.bus.subscribe(mid, websocket)
        try:
            m = hub.db.match(mid)
            if m:
                await websocket.send_json({"type": "state", "data": hub.match_state(m)})
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            await hub.bus.unsubscribe(mid, websocket)

    # -- QR
    @app.get("/api/qr.png")
    async def qr_png(url: str = Query(min_length=4, max_length=300)):
        img = qrcode.make(url, box_size=8, border=1)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    # -- web apps (HTML gets the build number injected for cache busting)
    def page(name: str):
        async def handler():
            html = (WEB_DIR / name).read_text(encoding="utf-8")
            html = html.replace("{{BUILD}}", str(version.BUILD_NUMBER)).replace("{{SEMVER}}", version.SEMVER)
            return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
        return handler

    for name in ("index.html", "camera.html", "viewer.html", "lobby.html", "calibrate.html"):
        app.get(f"/{name}", include_in_schema=False)(page(name))
    app.get("/", include_in_schema=False)(page("index.html"))

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest():
        return FileResponse(WEB_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")

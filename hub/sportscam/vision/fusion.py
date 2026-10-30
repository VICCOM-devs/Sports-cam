"""Multi-camera fusion and skating stats.

Each camera contributes (rink x, y, team, camera track id) per tick. Players are fused by nearest
neighbour within a gate (1.5 m) and smoothed with an exponential filter. Speed is derived from
the smoothed path with a short window, which keeps the number honest (no 60 km/h glitches).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Observation:
    cam_id: str
    track_id: int
    x: float
    y: float
    team: str | None
    t: float


@dataclass
class Player:
    id: int
    x: float
    y: float
    t: float
    team: str | None = None
    sources: dict[str, int] = field(default_factory=dict)   # cam_id → track id
    misses: int = 0
    path: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y)
    distance_m: float = 0.0
    top_speed_kmh: float = 0.0
    speed_kmh: float = 0.0
    label: str | None = None
    opted_out: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2), "team": self.team,
                "speed_kmh": round(self.speed_kmh, 1), "top_speed_kmh": round(self.top_speed_kmh, 1),
                "distance_m": round(self.distance_m), "label": self.label, "cams": len(self.sources)}


class Fusion:
    def __init__(self, gate_m: float = 1.5, alpha: float = 0.5, max_misses: int = 15, rink=None):
        self.gate = gate_m
        self.alpha = alpha
        self.max_misses = max_misses
        self.rink = rink
        self.players: dict[int, Player] = {}
        self._next = 1

    def _on_ice(self, x: float, y: float) -> bool:
        if self.rink is None:
            return True
        return abs(x) <= self.rink.length / 2 + 1 and abs(y) <= self.rink.width / 2 + 1

    def step(self, obs: list[Observation], t: float) -> list[Player]:
        obs = [o for o in obs if self._on_ice(o.x, o.y)]
        alive = {(o.cam_id, o.track_id) for o in obs}
        # 1. observations from a camera track already linked to a player go straight there
        assigned: dict[int, list[Observation]] = {}
        free: list[Observation] = []
        for o in obs:
            pid = next((p.id for p in self.players.values() if p.sources.get(o.cam_id) == o.track_id), None)
            if pid is not None and math.hypot(self.players[pid].x - o.x, self.players[pid].y - o.y) < self.gate * 2:
                assigned.setdefault(pid, []).append(o)
            else:
                free.append(o)
        # 2. free observations: nearest unassigned player within the gate, else a new player
        for o in sorted(free, key=lambda o: o.t):
            best, bd = None, self.gate
            for p in self.players.values():
                # a camera can only contribute one live track per player; a dead track may be replaced
                if o.cam_id in p.sources and p.sources[o.cam_id] != o.track_id and (o.cam_id, p.sources[o.cam_id]) in alive:
                    continue
                d = math.hypot(p.x - o.x, p.y - o.y)
                if d < bd:
                    best, bd = p, d
            if best is None:
                best = Player(self._next, o.x, o.y, t); self._next += 1
                self.players[best.id] = best
            best.sources[o.cam_id] = o.track_id
            assigned.setdefault(best.id, []).append(o)
        # 3. update positions, speeds, distances
        for p in list(self.players.values()):
            os_ = assigned.get(p.id)
            if not os_:
                p.misses += 1
                if p.misses > self.max_misses:
                    del self.players[p.id]
                continue
            p.misses = 0
            mx = sum(o.x for o in os_) / len(os_); my = sum(o.y for o in os_) / len(os_)
            nx, ny = p.x + self.alpha * (mx - p.x), p.y + self.alpha * (my - p.y)
            teams = [o.team for o in os_ if o.team]
            if teams:
                p.team = max(set(teams), key=teams.count)
            p.path.append((t, nx, ny))
            if len(p.path) > 3000:
                del p.path[:-3000]
            if len(p.path) >= 2:
                t0, x0, y0 = p.path[-2]
                d = math.hypot(nx - x0, ny - y0)
                if 0 < t - t0 < 2:
                    p.distance_m += d
                    # speed over the last ~1 s window
                    win = [q for q in p.path if t - q[0] <= 1.0]
                    if len(win) >= 2 and win[-1][0] - win[0][0] > 0.3:
                        dist = sum(math.hypot(win[i][1] - win[i - 1][1], win[i][2] - win[i - 1][2]) for i in range(1, len(win)))
                        p.speed_kmh = min(dist / (win[-1][0] - win[0][0]) * 3.6, 45.0)
                        p.top_speed_kmh = max(p.top_speed_kmh, p.speed_kmh)
            p.x, p.y, p.t = nx, ny, t
        return [p for p in self.players.values() if p.misses == 0]

    def snapshot(self) -> list[dict]:
        return [p.to_dict() for p in self.players.values() if p.misses <= 2 and not p.opted_out]

    def stats(self) -> list[dict]:
        out = []
        for p in self.players.values():
            if len(p.path) < 10:
                continue
            out.append({"id": p.id, "label": p.label, "team": p.team, "distance_m": round(p.distance_m),
                        "top_speed_kmh": round(p.top_speed_kmh, 1),
                        "time_tracked_s": round(p.path[-1][0] - p.path[0][0]),
                        "path": [(round(t, 2), round(x, 2), round(y, 2)) for t, x, y in p.path[::2]]})
        return out

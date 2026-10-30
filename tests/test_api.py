"""API tests against the built-in media engine (no network, no browser)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hub"))

from sportscam.config import Config  # noqa: E402
from sportscam.main import create_app  # noqa: E402
from sportscam.updates import UpdateChecker  # noqa: E402


@pytest.fixture
def client(tmp_path):
    cfg = Config(data_dir=tmp_path, media_engine="pyrtc", update_url="http://127.0.0.1:1/none.json")
    with TestClient(create_app(cfg)) as c:
        yield c


def test_version_and_update_state(client):
    v = client.get("/api/version").json()
    assert v["build"] >= 1 and v["semver"] and v["media"] == "pyrtc"
    assert v["update"]["status"] in ("unknown", "unavailable")  # offline never blocks


def test_match_flow(client):
    r = client.post("/api/matches", json={"home": "Frölunda U13", "away": "Färjestad U13"})
    assert r.status_code == 201, r.text
    created = r.json()
    code = created["match"]["code"]
    assert len(code) == 6 and created["join_url"].endswith(f"?c={code}")
    admin = {"X-Token": created["admin_token"]}

    # viewer joins with the code, lower-case and with a space is fine
    j = client.post("/api/join", json={"code": code.lower()[:3] + " " + code[3:], "role": "viewer"}).json()
    viewer = {"X-Token": j["token"]}
    assert j["match"]["home"] == "Frölunda U13"

    # viewer may tag a highlight but not a goal
    assert client.post("/api/events", json={"kind": "highlight"}, headers=viewer).status_code == 201
    assert client.post("/api/events", json={"kind": "goal", "team": "home"}, headers=viewer).status_code == 403

    # scorekeeper runs clock and score
    sk = {"X-Token": client.post("/api/join", json={"code": code, "role": "scorekeeper"}).json()["token"]}
    assert client.post("/api/match/start", headers=sk).json()["status"] == "live"
    assert client.post("/api/match/clock", json={"running": True}, headers=sk).json()["clock_running"] is True
    st = client.post("/api/match/score", json={"home_score": 1}, headers=sk).json()
    assert st["home_score"] == 1
    assert client.post("/api/events", json={"kind": "goal", "team": "home"}, headers=sk).status_code == 201

    # camera registers and shows up in state
    cam_tok = {"X-Token": client.post("/api/join", json={"code": code, "role": "camera", "label": "Anna"}).json()["token"]}
    cam = client.post("/api/cameras", json={"label": "Corner A"}, headers=cam_tok).json()
    assert cam["whip"].startswith("/whip/")
    state = client.get("/api/match", headers=viewer).json()
    assert [c["label"] for c in state["cameras"]] == ["Corner A"]
    assert {e["kind"] for e in state["events"]} >= {"highlight", "score", "goal"}

    # viewer cannot play a camera that is not publishing
    assert client.post(f"/whep/{cam['camera_id']}", content="v=0", headers=viewer).status_code == 404

    # only admin ends the match
    assert client.post("/api/match/end", headers=sk).status_code == 403
    assert client.post("/api/match/end", headers=admin).json()["status"] == "ended"


def test_bad_code_and_token(client):
    assert client.post("/api/join", json={"code": "ZZZZZZ"}).status_code == 404
    assert client.get("/api/match", headers={"X-Token": "nope"}).status_code == 401


def test_update_gate_logic():
    assert UpdateChecker.evaluate({"min_version_code": 1, "latest_version_code": 1}, build=1)["status"] == "current"
    assert UpdateChecker.evaluate({"min_version_code": 1, "latest_version_code": 3}, build=1)["status"] == "nudge"
    assert UpdateChecker.evaluate({"min_version_code": 2, "latest_version_code": 3}, build=1)["status"] == "blocked"
    assert UpdateChecker.evaluate({"latest_version_code": 3, "force": True}, build=2)["status"] == "blocked"
    assert UpdateChecker.evaluate({}, build=1)["status"] == "current"

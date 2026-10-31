"""End to end in a real browser: lobby creates a match, a fake camera publishes over WHIP,
a viewer receives frames over WHEP, a tag reaches the viewer's event list. Also saves screenshots
for the UI review pass."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "tests" / "screenshots"
MODELS = Path("/home/claude/downloads/models")
FAKE_VIDEO = Path("/tmp/anysport-fake-cam.y4m")


def _fake_video() -> Path:
    """A slow pan over a photo with two people, so the fake camera produces real detections."""
    if not FAKE_VIDEO.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(ROOT / "tests/assets/zidane.jpg"),
                        "-t", "6", "-r", "10", "-vf", "crop=1000:560:x='min(t*40,280)':y=80,scale=640:360",
                        "-pix_fmt", "yuv420p", str(FAKE_VIDEO)], check=True)
    return FAKE_VIDEO


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    data = tmp_path_factory.mktemp("hubdata")
    http, https = _free_port(), _free_port()
    env = {**os.environ, "ANYSPORT_DATA": str(data), "ANYSPORT_HTTP_PORT": str(http),
           "ANYSPORT_HTTPS_PORT": str(https), "ANYSPORT_HOST": "127.0.0.1",
           "ANYSPORT_MEDIA": "pyrtc", "ANYSPORT_UPDATE_URL": "http://127.0.0.1:1/none.json",
           "ANYSPORT_MODELS": str(MODELS), "ANYSPORT_VISION_FPS": "5",
           "PYTHONPATH": str(ROOT / "hub")}
    proc = subprocess.Popen([sys.executable, "-m", "anysport"], env=env, cwd=ROOT / "hub")
    base = f"http://127.0.0.1:{http}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/api/version", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill(); raise RuntimeError("hub did not start")
    yield {"base": base, "data": data}
    proc.terminate(); proc.wait(timeout=10)


def test_full_match_flow(hub):
    SHOTS.mkdir(exist_ok=True)
    base = hub["base"]
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                          f"--use-file-for-fake-video-capture={_fake_video()}",
                                          "--autoplay-policy=no-user-gesture-required"])
        # --- lobby on the hub PC
        lobby = browser.new_context(viewport={"width": 1280, "height": 860}, locale="sv-SE").new_page()
        lobby.goto(f"{base}/lobby.html"); lobby.wait_for_selector("body[data-ready]")
        lobby.fill("#home", "Frölunda HC U13")
        lobby.fill("#away", "Färjestad BK U13")
        lobby.click("#form button")
        lobby.wait_for_selector("#code")
        code = lobby.inner_text("#code").strip()
        assert len(code) == 6
        lobby.screenshot(path=str(SHOTS / "lobby.png"))

        # --- camera phone (fake device → WHIP)
        cam_ctx = browser.new_context(viewport={"width": 390, "height": 844}, locale="sv-SE",
                                      permissions=["camera", "microphone"])
        cam = cam_ctx.new_page()
        cam.goto(f"{base}/?c={code}"); cam.wait_for_selector("body[data-ready]")
        cam.click('button[data-role="camera"]')
        cam.wait_for_url("**/camera.html*")
        cam.wait_for_selector("#start")
        cam.screenshot(path=str(SHOTS / "camera-setup.png"))
        cam.click("#start")
        cam.wait_for_function("!document.getElementById('live').classList.contains('hidden') && document.getElementById('state').classList.contains('live')", timeout=15000)
        cam.screenshot(path=str(SHOTS / "camera-live.png"))

        # --- scorekeeper starts the game (so recording kicks in)
        sk = browser.new_context(viewport={"width": 390, "height": 844}, locale="sv-SE").new_page()
        sk.goto(f"{base}/?c={code}"); sk.wait_for_selector("body[data-ready]")
        sk.click('button[data-role="scorekeeper"]')
        sk.wait_for_selector("#startgame:not(.hidden)")
        sk.click("#startgame")
        sk.wait_for_selector("#livebadge:not(.hidden)")
        sk.click("#clockbtn")
        sk.click("#plus-h")
        sk.screenshot(path=str(SHOTS / "scorekeeper.png"))

        # --- parent viewer (WHEP) sees real frames and the goal
        viewer = browser.new_context(viewport={"width": 390, "height": 844}, locale="en-US").new_page()
        viewer.goto(f"{base}/?c={code}"); viewer.wait_for_selector("body[data-ready]")
        viewer.click('button[data-role="viewer"]')
        viewer.wait_for_url("**/viewer.html*")
        viewer.wait_for_function(
            "[...document.querySelectorAll('video')].some(v => v.classList.contains('on') && v.videoWidth > 0)",
            timeout=20000)
        assert viewer.inner_text("#score-h").strip() == "1"
        viewer.click(".tag.primary")  # Highlight
        viewer.wait_for_selector(".event")
        viewer.screenshot(path=str(SHOTS / "viewer.png"))
        assert "Highlight" in viewer.inner_text("#eventlist")

        # lobby shows the camera as live and the viewer count
        lobby.wait_for_function("document.querySelector('#cams .chip.live') !== null", timeout=10000)
        lobby.screenshot(path=str(SHOTS / "lobby-live.png"))

        # --- calibration page: snapshot loads, tap 5 landmarks, save → overlay drawn
        cal_href = lobby.get_attribute("#cams a.btn", "href")
        cal = browser.new_context(viewport={"width": 1100, "height": 800}, locale="sv-SE").new_page()
        cal.goto(f"{base}{cal_href}"); cal.wait_for_selector("body[data-ready]")
        cal.wait_for_function("document.getElementById('snap').naturalWidth > 0", timeout=15000)
        box = cal.locator("#snap").bounding_box()
        # synthetic "camera": rink metres → image pixels (a plausible corner view)
        import numpy as np
        H = np.array([[10.0, 0.0, 320.0], [0.0, -20.0, 180.0], [0.0, 0.0, 1.0]])
        rink = httpx.get(f"{base}/api/rink").json()
        lms = {l["id"]: l for l in rink["landmarks"]}
        for lid in ("center", "dot_home_L", "dot_home_R", "dot_away_L", "dot_away_R", "ndot_home_L"):
            cal.locator(f'#rink circle[data-id="{lid}"]').click(force=True)
            v = H @ np.array([lms[lid]["x"], lms[lid]["y"], 1.0])
            px, py = v[0] / v[2], v[1] / v[2]
            cal.mouse.click(box["x"] + px / 640 * box["width"], box["y"] + py / 360 * box["height"])
        cal.click("#save")
        cal.wait_for_function("document.querySelector('#overlay .ov') !== null", timeout=10000)
        cal.screenshot(path=str(SHOTS / "calibrate.png"))

        # --- the rink's own camera joins as an external source (a file stands in for RTSP here)
        lobby.click("details summary")
        lobby.fill("#ext-label", "Rinkkamera"); lobby.fill("#ext-url", str(_fake_video())); lobby.click("#extform button")
        lobby.wait_for_function("document.querySelectorAll('#cams .cam').length === 2", timeout=10000)
        viewer.wait_for_function("document.querySelectorAll('.angle:not(.auto):not(.dead)').length === 2", timeout=15000)
        viewer.wait_for_selector(".angle.auto", timeout=5000)
        viewer.click(".angle:not(.auto) >> nth=1")
        viewer.wait_for_function("document.getElementById('anglename').textContent === 'Rinkkamera'", timeout=15000)
        viewer.click('#tabs [data-pane="live"]')
        # broadcast overlay: rings under the detected people, a name tag once a roster name is linked
        viewer.wait_for_function("document.querySelectorAll('#ov ellipse.ring').length >= 1", timeout=20000)
        viewer.screenshot(path=str(SHOTS / "viewer-two-angles.png"))

        # --- vision: detections happen, players appear on the viewer's rink map
        viewer.click('#tabs [data-pane="rink"]')
        viewer.wait_for_function("document.querySelectorAll('#rinkmap circle.p').length > 0", timeout=30000)
        viewer.screenshot(path=str(SHOTS / "viewer-rink.png"))
        st = httpx.get(f"{base}/api/match", headers={"X-Token": lobby.evaluate("sessionStorage.getItem('anysport.admin')")}).json()
        assert st["vision"]["enabled"] and sum(c["detections"] for c in st["vision"]["cameras"].values()) > 0
        time.sleep(6)
        viewer.click('#tabs [data-pane="players"]')
        viewer.wait_for_selector("#players .prow", timeout=25000)
        viewer.screenshot(path=str(SHOTS / "viewer-players.png"))

        # --- scorekeeper manages the roster and opts a player out
        sk.click('#tabs [data-pane="players"]')
        sk.wait_for_selector("#rosterbox:not(.hidden)")
        sk.fill("#r-num", "17"); sk.fill("#r-name", "Elsa Lindqvist"); sk.click("#rosterform button")
        sk.wait_for_selector(".rrow")
        sk.click('.rrow [data-a="out"]')
        sk.wait_for_selector(".rrow.out")
        sk.screenshot(path=str(SHOTS / "scorekeeper-roster.png"))
        assert viewer.evaluate("fetch('/api/roster',{headers:{'X-Token':JSON.parse(localStorage.getItem('anysport.session')).token}}).then(r=>r.json()).then(j=>j.length)") == 0

        # --- link a named player to a tracked dot → name tag with speed appears in the overlay
        adm = {"X-Token": lobby.evaluate("sessionStorage.getItem('anysport.admin')")}
        rid = httpx.post(f"{base}/api/roster", json={"team": "away", "number": "9", "name": "Noa Berg"}, headers=adm).json()["id"]
        wide = browser.new_context(viewport={"width": 1100, "height": 700}, locale="sv-SE").new_page()
        wide.goto(f"{base}/viewer.html?c={code}&t={adm['X-Token']}"); wide.wait_for_selector("body[data-ready]")
        wide.wait_for_function("document.querySelectorAll('#ov ellipse.ring').length >= 1", timeout=20000)
        wide.click("#zoom")  # full frame for the screenshot
        # the looping test video makes tracked ids churn, so link to whoever is on screen right now, retrying
        for _ in range(5):
            players = httpx.get(f"{base}/api/positions", headers=adm).json()["players"]
            if players:
                httpx.patch(f"{base}/api/roster/{rid}", json={"player_id": players[0]["id"]}, headers=adm)
            try:
                wide.wait_for_function("document.querySelector('#ov .tag') !== null", timeout=4000)
                break
            except Exception:
                continue
        else:
            raise AssertionError("name tag never appeared in the overlay")
        time.sleep(0.6)
        wide.screenshot(path=str(SHOTS / "viewer-overlay.png"))

        # --- end match from lobby; recording file exists
        lobby.click("#end")
        viewer.wait_for_function("document.querySelector('#livebadge').classList.contains('hidden')", timeout=10000)
        time.sleep(1.5)
        recs = list((hub["data"] / "recordings" / code).glob("*.mkv"))
        assert recs and recs[0].stat().st_size > 10_000, recs
        tracking = hub["data"] / "recordings" / code / "tracking.json"
        assert tracking.exists() and tracking.stat().st_size > 50

        # --- clips were cut from the tags (goal + highlight) and play in the viewer
        viewer.click('#tabs [data-pane="clips"]')
        viewer.wait_for_function("document.querySelectorAll('#clips .clip').length >= 1", timeout=30000)
        viewer.wait_for_function("[...document.querySelectorAll('#clips video')].some(v => v.readyState >= 1)", timeout=15000)
        viewer.screenshot(path=str(SHOTS / "viewer-clips.png"))
        clips = list((hub["data"] / "recordings" / code / "clips").glob("*.mp4"))
        assert clips and all(c.stat().st_size > 1000 for c in clips)

        # --- labelling tool on the hub: frame loads with suggestions, save writes YOLO txt + jpg
        lab = browser.new_context(viewport={"width": 1200, "height": 800}, locale="en-US").new_page()
        lab.goto(f"{base}/label.html"); lab.wait_for_selector("body[data-ready]")
        lab.wait_for_function("document.getElementById('img').naturalWidth > 0", timeout=15000)
        lab.wait_for_function("document.querySelectorAll('#ov .bx').length >= 1", timeout=10000)
        lab.click("#accept"); lab.click("#save")
        lab.wait_for_function("document.getElementById('count').textContent === '1'", timeout=10000)
        lab.screenshot(path=str(SHOTS / "label.png"))
        labels = list((hub["data"] / "recordings" / code / "labels").glob("*.txt"))
        assert labels and labels[0].read_text().strip().startswith("0 ")
        browser.close()

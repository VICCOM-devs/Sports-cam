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
FAKE_VIDEO = Path("/tmp/sportscam-fake-cam.y4m")


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
    env = {**os.environ, "SPORTSCAM_DATA": str(data), "SPORTSCAM_HTTP_PORT": str(http),
           "SPORTSCAM_HTTPS_PORT": str(https), "SPORTSCAM_HOST": "127.0.0.1",
           "SPORTSCAM_MEDIA": "pyrtc", "SPORTSCAM_UPDATE_URL": "http://127.0.0.1:1/none.json",
           "SPORTSCAM_MODELS": str(MODELS), "SPORTSCAM_VISION_FPS": "5",
           "PYTHONPATH": str(ROOT / "hub")}
    proc = subprocess.Popen([sys.executable, "-m", "sportscam"], env=env, cwd=ROOT / "hub")
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
        lobby.goto(f"{base}/lobby.html")
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
        cam.goto(f"{base}/?c={code}")
        cam.click('button[data-role="camera"]')
        cam.wait_for_url("**/camera.html*")
        cam.wait_for_selector("#start")
        cam.screenshot(path=str(SHOTS / "camera-setup.png"))
        cam.click("#start")
        cam.wait_for_function("!document.getElementById('live').classList.contains('hidden') && document.getElementById('state').classList.contains('live')", timeout=15000)
        cam.screenshot(path=str(SHOTS / "camera-live.png"))

        # --- scorekeeper starts the game (so recording kicks in)
        sk = browser.new_context(viewport={"width": 390, "height": 844}, locale="sv-SE").new_page()
        sk.goto(f"{base}/?c={code}")
        sk.click('button[data-role="scorekeeper"]')
        sk.wait_for_selector("#startgame:not(.hidden)")
        sk.click("#startgame")
        sk.wait_for_selector("#livebadge:not(.hidden)")
        sk.click("#clockbtn")
        sk.click("#plus-h")
        sk.screenshot(path=str(SHOTS / "scorekeeper.png"))

        # --- parent viewer (WHEP) sees real frames and the goal
        viewer = browser.new_context(viewport={"width": 390, "height": 844}, locale="en-US").new_page()
        viewer.goto(f"{base}/?c={code}")
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
        cal.goto(f"{base}{cal_href}")
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

        # --- vision: detections happen, players appear on the viewer's rink map
        viewer.click('#tabs [data-pane="rink"]')
        viewer.wait_for_function("document.querySelectorAll('#rinkmap circle.p').length > 0", timeout=30000)
        viewer.screenshot(path=str(SHOTS / "viewer-rink.png"))
        st = httpx.get(f"{base}/api/match", headers={"X-Token": lobby.evaluate("sessionStorage.getItem('sportscam.admin')")}).json()
        assert st["vision"]["enabled"] and sum(c["detections"] for c in st["vision"]["cameras"].values()) > 0
        time.sleep(6)
        viewer.click('#tabs [data-pane="players"]')
        viewer.wait_for_selector("#players .prow", timeout=25000)
        viewer.screenshot(path=str(SHOTS / "viewer-players.png"))

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
        browser.close()

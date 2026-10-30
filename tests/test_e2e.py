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


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    data = tmp_path_factory.mktemp("hubdata")
    http, https = _free_port(), _free_port()
    env = {**os.environ, "SPORTSCAM_DATA": str(data), "SPORTSCAM_HTTP_PORT": str(http),
           "SPORTSCAM_HTTPS_PORT": str(https), "SPORTSCAM_HOST": "127.0.0.1",
           "SPORTSCAM_MEDIA": "pyrtc", "SPORTSCAM_UPDATE_URL": "http://127.0.0.1:1/none.json",
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

        # --- end match from lobby; recording file exists
        lobby.click("#end")
        viewer.wait_for_function("document.querySelector('#livebadge').classList.contains('hidden')", timeout=10000)
        time.sleep(1.5)
        recs = list((hub["data"] / "recordings" / code).glob("*.mp4"))
        assert recs and recs[0].stat().st_size > 10_000, recs
        browser.close()

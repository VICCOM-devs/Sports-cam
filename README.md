# VICCOM Sports Cam (working name)

Multi-phone sports camera system for small clubs. Parents' phones film from around the rink, a
laptop at the rink is the hub, anyone with the match code watches live from any angle. Ice hockey
first. Spec: *Phase 1 Spec – Multi-Phone Sports Camera System* (Claude Doc in the project).

**This is milestone M1:** capture + QR join + multi-angle live viewing + score/clock + tags +
recording. No tracking yet (M2+).

## Run the hub (Windows)

1. Install Python 3.11+ from python.org (tick "Add to PATH").
2. Double-click `run-hub.bat`. First start creates a virtualenv and installs dependencies.
3. Open `http://localhost:8080/lobby.html` on the laptop, create a game, show the QR code.
4. Phones on the same Wi-Fi scan the code. "Titta" = watch, "Jag filmar" = camera.

macOS/Linux: `./run-hub.sh`.

The hub listens on **http://\<lan-ip\>:8080** (viewers, lobby) and **https://\<lan-ip\>:8443**
(camera phones; browsers only allow the camera on https). The https certificate is self-signed:
the first time a camera phone opens it, tap *Advanced → Proceed* once.

Environment variables: `SPORTSCAM_DATA` (data dir), `SPORTSCAM_HTTP_PORT`, `SPORTSCAM_HTTPS_PORT`,
`SPORTSCAM_HOST` (LAN IP for the QR code), `SPORTSCAM_MEDIA` (`auto` | `pyrtc` | `mediamtx`),
`SPORTSCAM_RECORD` (`0` to disable), `SPORTSCAM_UPDATE_URL`.

## Media engines

| Engine | When | Notes |
|---|---|---|
| `pyrtc` (built-in, aiortc) | default until MediaMTX is present; tests | re-encodes per viewer, fine for 1–3 cameras |
| `mediamtx` | real games | hub downloads MediaMTX v1.15.3 for the OS on first run, verifies the SHA-256 from the release, runs it as a child process; WHIP/WHEP are proxied; recordings are fMP4 segments |

Switch with `SPORTSCAM_MEDIA=mediamtx` (downloads on first start; needs internet once).

## Layout

```
hub/sportscam/      FastAPI hub: API, WebSocket events, WHIP/WHEP signalling, static web apps
  media/            pyrtc.py (aiortc) · mediamtx.py (process + proxy)
  updates.py        version gate (viccom-auto-update)  ·  certs.py  self-signed https
web/                index (join) · camera · viewer · lobby  — vanilla ES modules, no build step
  css/tokens.css    design tokens (design-system/MASTER.md)
  i18n/             sv.json, en.json
tests/              pytest API tests + Playwright end-to-end (fake camera → viewer frames)
docs/updates/       sports-cam.json → copy to VICCOM_Site/updates/
```

## Tests

```
pip install -r hub/requirements.txt pytest playwright
python -m pytest tests -q
```

The e2e test starts a real hub, publishes Chromium's fake camera over WHIP, plays it back over
WHEP in a viewer, scores a goal, tags a highlight, ends the game and checks the MP4 on disk.
Screenshots land in `tests/screenshots/`.

## Roles and access

| Role | How they get in | Can |
|---|---|---|
| viewer | match code | watch any angle, tag Highlight |
| scorekeeper | match code + "I'm keeping score" | start game, clock, score, period, all tags |
| camera | match code + "I'm filming" | register a camera, publish |
| admin | created with the match (token in the lobby's "admin view" link) | everything + end game |

Match creation is only accepted from the hub machine itself (localhost).

## Updates

Layer 1 version gate: the hub fetches `https://viccom.se/updates/sports-cam.json` on start and
every 30 min; `/api/version` exposes the result; the web apps show a nudge or a blocking screen.
Offline never blocks. Web assets are versioned with the hub build, so a new hub build shows a
"New version – reload" toast on open pages. To ship: bump `hub/sportscam/version.py`
(SEMVER + BUILD_NUMBER), publish the build, then raise `latest_version_code` in the JSON.

# Sports Cam – handover

Read `viccom-studio` first. Spec of record: Claude Doc *Phase 1 Spec – Multi-Phone Sports Camera
System (Ice Hockey)* in the "Camera Project For Sports" project.

## State (2026-10-30, afternoon)

- **Repo**: `VICCOM-devs/sports-cam`, branch `main`. Everything below is pushed.
- **M1–M3 built and tested** (12 tests, e2e in a real browser, stable over 6 consecutive runs):
  capture + QR join + viewer (M1); detection (YOLOX nano, ONNX, Apache-2.0), per-camera tracker,
  rink calibration page, jersey-colour teams, multi-camera fusion, live rink map, distance/top
  speed (M2); auto director with hysteresis, follow-cam crop ("Följ spelet"), highlight clips cut
  by ffmpeg around tags, Clips tab, roster with names/numbers, link dot→player, per-player
  opt-out, external camera sources (RTSP/file) as extra angles on both engines (M3 parts).
- **Recording** is now an ffmpeg pipe (video only, x264 veryfast, .mkv) because aiortc's
  MediaRecorder produced empty files intermittently. Audio recording: TODO (needed for crowd-noise
  highlights).
- Not done from the spec: Active Play (cut stoppages), spotlight reel per player, telestration,
  instant replay scrubbing (viewer shows live only), tournament mode, native camera app.

## Earlier state

- **Version** 0.1.0, build 1, channel alpha. Working name "Sports Cam" (`se.viccom.sportscam`
  reserved as package id). Product name not decided.
- **Milestone M1 done in code**: hub (FastAPI + aiortc/MediaMTX), QR/code join, browser camera
  page publishing over WHIP, viewer with angle switching over WHEP, scoreboard + game clock,
  tags (Goal/Shot/Save/Hit/Penalty/Highlight), WebSocket live state, recording to MP4, lobby with
  QR and camera health grid, SV/EN, design tokens, auto-update version gate.
- **Tests green**: 4 API tests + 1 Playwright end-to-end (fake camera → viewer frames → tag →
  recording file). `python -m pytest tests -q` ≈ 9 s.
- **Not yet tested on real hardware**: no real phone or rink yet. The https self-signed step on
  phones and Wi-Fi behaviour are the first things to try.
- **Repo**: none yet. Code is in Claude's workspace at `/home/claude/sports-cam` with local git
  history. Victor creates `VICCOM-devs/sports-cam`; then push.

## Decisions

- Phones first, native camera app later. The camera page is a web app so iPhones work on day one
  and nothing needs installing. A Compose Multiplatform camera app (reusing AnyLap's camera code)
  is the plan once the web version shows its limits (background, heat, codec control).
- Two media engines behind one interface. aiortc built in (no binary, tests) and MediaMTX for real
  games (downloaded + checksum-verified on first run on Victor's PC). The workspace cannot run
  downloaded binaries, so MediaMTX is only unit-level tested here.
- Match creation is localhost-only (the lobby runs on the hub). Everyone else joins by code with
  a role; admin token is minted with the match.
- Viewers use http; cameras use https (secure origin needed for getUserMedia). Viewer page
  hops to the https URL from `/api/version.secure_url` when the camera role is picked.
- No face recognition ever; privacy section removed from the spec at Victor's request, but a
  per-player opt-out stays in the feature list (M3).
- Rink camera ingest (Solidsport/Spiideo feeds, RTSP/HDMI) is in the spec; not started.

## Next steps (M2)

1. Push to the repo, run on Victor's Windows PC with two real phones on a home router. Fix what
   breaks (expect: cert prompt wording, autoplay/unmute, battery dimmer).
2. `SPORTSCAM_MEDIA=mediamtx` on Windows: confirm download, checksum, WHIP via proxy, fMP4 recording.
3. Rink calibration UI (tap landmarks) + camera → rink homography.
4. Player detection on recorded clips (Apache-licensed detector; see spec risk on YOLO/AGPL).
5. Referral button on the join page already links viccom.se; make a real landing page later.

## How to push an update

Bump `hub/sportscam/version.py` (SEMVER, BUILD_NUMBER+1), tag, build, then raise
`latest_version_code` in `VICCOM_Site/updates/sports-cam.json` (copy from `docs/updates/`).
Raise `min_version_code` only for breaking protocol changes, after asking Victor.

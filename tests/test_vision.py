"""Vision unit tests: detector on real photos, calibration round-trip, tracker identity, fusion stats."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hub"))

from anysport.vision.detector import Detection, PersonDetector  # noqa: E402
from anysport.vision.fusion import Fusion, Observation  # noqa: E402
from anysport.vision.rink import TEMPLATES, calibrate  # noqa: E402
from anysport.vision.teams import TeamClassifier, torso_feature  # noqa: E402
from anysport.vision.tracker import Tracker  # noqa: E402

MODELS = Path("/home/claude/downloads/models")


@pytest.mark.skipif(not (MODELS / "yolox_nano.onnx").exists(), reason="model not downloaded")
def test_detector_finds_people():
    det = PersonDetector(MODELS / "yolox_nano.onnx", 416)
    img = cv2.imread(str(ROOT / "tests/assets/zidane.jpg"))
    people = det(img)
    assert len(people) == 2
    assert all(p.score > 0.5 for p in people)


def test_calibration_round_trip():
    rink = TEMPLATES["iihf"]
    # synthetic camera: a known homography from rink metres to pixels
    H_true = np.array([[18.0, -4.0, 640.0], [2.0, 9.0, 300.0], [0.0, 0.004, 1.0]])
    pairs = []
    for lid in ("center", "dot_home_L", "dot_home_R", "blue_home_L", "blue_home_R", "ndot_home_L"):
        lm = next(l for l in rink.landmarks if l.id == lid)
        v = H_true @ np.array([lm.x, lm.y, 1.0])
        pairs.append({"id": lid, "px": v[0] / v[2] + np.random.uniform(-1, 1), "py": v[1] / v[2] + np.random.uniform(-1, 1)})
    cal = calibrate(rink, pairs)
    assert cal.rms_error_m < 0.5
    # an unseen point maps back within half a metre
    v = H_true @ np.array([-15.0, 3.0, 1.0])
    x, y = cal.image_to_rink(v[0] / v[2], v[1] / v[2])
    assert math.hypot(x + 15, y - 3) < 0.5
    with pytest.raises(ValueError):
        calibrate(rink, pairs[:3])


def test_tracker_keeps_identity_through_a_gap():
    tr = Tracker(dt=0.2, max_misses=5)
    ids = []
    for i in range(20):
        dets = [] if 8 <= i < 11 else [Detection(100 + i * 6, 200, 140 + i * 6, 300, 0.9)]
        out = tr.step(dets)
        if out:
            ids.append(out[0].id)
    assert len(set(ids)) == 1, ids


def test_team_feature_and_clustering():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :150] = (200, 40, 40)   # blue-ish jerseys (BGR)
    img[:, 150:] = (30, 30, 220)   # red jerseys
    feats = [torso_feature(img, (10 + (i % 5) * 20, 20, 50 + (i % 5) * 20, 180)) for i in range(5)]
    feats += [torso_feature(img, (160 + (i % 5) * 20, 20, 200 + (i % 5) * 20, 180)) for i in range(5)]
    tc = TeamClassifier()
    assert tc.fit(feats)
    teams = [tc.classify(f) for f in feats]
    assert len(set(teams[:5])) == 1 and len(set(teams[5:])) == 1 and teams[0] != teams[5]
    grey = np.full((200, 100, 3), 128, np.uint8)
    assert tc.classify(torso_feature(grey, (10, 10, 90, 190))) == "ref"


def test_fusion_merges_cameras_and_measures_speed():
    f = Fusion(rink=TEMPLATES["iihf"])
    # one skater seen by two cameras, moving 5 m/s along x
    for k in range(25):
        t = k * 0.2
        x = -10 + 5 * t
        obs = [Observation("camA", 1, x + 0.2, 2.0, "home", t), Observation("camB", 7, x - 0.2, 2.1, "home", t)]
        players = f.step(obs, t)
    assert len(f.players) == 1
    p = players[0]
    assert len(p.sources) == 2
    assert 15 < p.speed_kmh < 21        # 5 m/s = 18 km/h
    assert 20 < p.distance_m < 26       # 4.8 s × 5 m/s
    assert p.team == "home"
    # an observation far outside the rink is ignored
    f.step([Observation("camA", 2, 80, 80, None, 6.0)], 6.0)
    assert len(f.players) == 1


def test_custom_model_loads_and_reports_classes(tmp_path):
    """A hub-exported multi-class model (from ml/) loads through the same detector and tags classes."""
    import shutil
    src = Path("/tmp/anysport-test.onnx")
    if not src.exists():
        pytest.skip("no exported test model")
    shutil.copy(src, tmp_path / "custom.onnx"); shutil.copy(src.with_suffix(".json"), tmp_path / "custom.json")
    det = PersonDetector.load(tmp_path, "nano", conf=0.2)
    assert det.name == "custom" and det.classes[3] == "puck"
    import subprocess, sys as _s
    subprocess.run([_s.executable, str(ROOT / "ml/make_synthetic.py"), "--out", str(tmp_path / "syn"), "--n", "2", "--size", "320"], check=True, cwd=ROOT / "ml")
    img = cv2.imread(str(tmp_path / "syn/images/syn_0000.jpg"))
    dets = det(img)
    assert dets and {d.cls for d in dets} & {"player", "goal", "faceoff_dot"}

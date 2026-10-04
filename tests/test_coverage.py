import numpy as np
from types import SimpleNamespace

from galnp_s.capture.coverage_report import gate_g0, _status, SUPPORTED_MIN_FRAMES


def test_status_thresholds():
    assert _status(SUPPORTED_MIN_FRAMES, 0.5) == "supported"
    assert _status(SUPPORTED_MIN_FRAMES, 0.2) == "partial"
    assert _status(5, 0.9) == "unsupported"


def _mk_report(n_support=40, critical_ok=True, yaw_pos=25, yaw_neg=25,
               pitch_pos=15, pitch_neg=15, blur=0.9, color_warn=0.0):
    return {
        "n_channels_supported": n_support,
        "n_channels_partial": 0,
        "n_channels_unsupported": 0,
        "critical_channels_ok": critical_ok,
        "channels": [],
        "groups": {},
        "pose": {
            "yaw":   {"p95_pos": yaw_pos,   "p95_neg": yaw_neg},
            "pitch": {"p95_pos": pitch_pos, "p95_neg": pitch_neg},
            "roll":  {"p95_pos": 10, "p95_neg": 10},
        },
        "gaze_p99_rad": {},
        "combinations": [],
        "frames": {
            "n_kept": 1000, "n_train": 800, "n_val": 100, "n_test": 100,
            "blur_pass_frac": blur, "color_gain_warn_frac": color_warn,
        },
        "color": {"gain_mean": 0.01, "gain_p95": 0.03},
        "pose_occupancy_yaw_pitch": [],
        "subject_id": "t",
    }


def test_g0_pass():
    cfg = SimpleNamespace()
    ok, fails = gate_g0(_mk_report(), cfg)
    assert ok and not fails


def test_g0_fails_on_critical():
    rep = _mk_report(critical_ok=False)
    ok, fails = gate_g0(rep, SimpleNamespace())
    assert not ok and any("critical" in m for m in fails)


def test_g0_fails_on_yaw():
    rep = _mk_report(yaw_pos=10, yaw_neg=25)
    ok, fails = gate_g0(rep, SimpleNamespace())
    assert not ok and any("yaw" in m for m in fails)


def test_g0_fails_on_blur():
    rep = _mk_report(blur=0.5)
    ok, fails = gate_g0(rep, SimpleNamespace())
    assert not ok and any("blur" in m for m in fails)
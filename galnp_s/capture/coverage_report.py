"""Coverage report (§11.4) and G0 gate (§10.6).

Reads `frames.npz`, produces `coverage.json` and (optionally) `coverage.html`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from galnp_s.capture import channels as C
from galnp_s.capture.prepare import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from galnp_s.utils.io import save_json
from galnp_s.utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED_MIN_FRAMES = 40
SUPPORTED_MIN_ACTIVE = 0.25
SUPPORTED_MIN_P95 = 0.40
PARTIAL_MIN_FRAMES = 10
PARTIAL_MIN_ACTIVE = 0.25


def _status(frames_above: int, p95: float) -> str:
    if frames_above >= SUPPORTED_MIN_FRAMES and p95 >= SUPPORTED_MIN_P95:
        return "supported"
    if frames_above >= PARTIAL_MIN_FRAMES:
        return "partial"
    return "unsupported"


def _percentile_signed(x: np.ndarray, side: str, p: float) -> float:
    if side == "pos":
        v = x[x > 0]
    else:
        v = -x[x < 0]
    return float(np.percentile(v, p)) if v.size else 0.0


def compute_coverage(frames_npz: str | Path, subject_id: str) -> dict:
    d = np.load(frames_npz)
    e = d["e"].astype(np.float32)
    pose = d["pose_deg"].astype(np.float32)
    gaze = d["gaze_rad"].astype(np.float32)
    split = d["split"]
    blur = d["blur"]
    gain = d["color_gain"]
    N = len(e)

    # ---- per-channel ----
    active = (e > SUPPORTED_MIN_ACTIVE).sum(axis=0)
    p95 = np.percentile(e, 95, axis=0)
    ch_rows = []
    for i, name in enumerate(C.CHANNEL_NAMES):
        if name == "_neutral":
            continue
        st = _status(int(active[i]), float(p95[i]))
        ch_rows.append({
            "name": name, "frames_above_0.25": int(active[i]),
            "p95": float(p95[i]), "status": st,
        })
    n_supported = sum(1 for r in ch_rows if r["status"] == "supported")
    n_partial = sum(1 for r in ch_rows if r["status"] == "partial")
    n_unsupported = sum(1 for r in ch_rows if r["status"] == "unsupported")
    critical_ok = all(
        any(r["name"] == c and r["status"] == "supported" for r in ch_rows)
        for c in C.CRITICAL_CHANNELS
    )

    # ---- per-group ----
    group_rows = {}
    for gname, members in (("U", C.UPPER_GROUP), ("L", C.LOWER_GROUP)):
        idxs = [C.NAME_TO_IDX[n] for n in members if n in C.NAME_TO_IDX]
        sub = e[:, idxs]
        group_rows[gname] = {
            "n_channels": len(idxs),
            "mean_activity": float(sub.mean()),
            "p95_activity": float(np.percentile(sub, 95)),
        }

    # ---- pose ----
    pose_rows = {}
    for j, axis in enumerate(("yaw", "pitch", "roll")):
        pose_rows[axis] = {
            "p01":  float(np.percentile(pose[:, j], 1)),
            "p50":  float(np.percentile(pose[:, j], 50)),
            "p95_pos": _percentile_signed(pose[:, j], "pos", 95),
            "p95_neg": _percentile_signed(pose[:, j], "neg", 95),
        }

    # 2-D yaw x pitch occupancy map (10 x 10 bins in +-40 deg)
    hist, _, _ = np.histogram2d(
        pose[:, 0], pose[:, 1],
        bins=[np.linspace(-40, 40, 11), np.linspace(-40, 40, 11)],
    )
    pose_occupancy = (hist / max(1, hist.sum())).tolist()

    # ---- gaze ----
    gaze_p99 = {}
    for k, tag in enumerate(("A_yaw", "A_pitch", "B_yaw", "B_pitch")):
        gaze_p99[tag] = float(np.percentile(np.abs(gaze[:, k]), 99)) if N else 0.0

    # ---- combinations ----
    pairs = []
    for gi, g in (("U", C.UPPER_GROUP), ("L", C.LOWER_GROUP)):
        idxs = [C.NAME_TO_IDX[n] for n in g if n in C.NAME_TO_IDX]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                ca = (e[:, ia] > 0.25) & (e[:, ib] > 0.25)
                if ca.sum() >= 150:
                    pairs.append({
                        "a": C.CHANNEL_NAMES[ia], "b": C.CHANNEL_NAMES[ib],
                        "group": gi, "co_active": int(ca.sum()),
                    })
    pairs.sort(key=lambda r: -r["co_active"])

    # ---- frames ----
    train_median_blur = float(np.median(blur[split == SPLIT_TRAIN])) if (split == SPLIT_TRAIN).any() else 0.0
    pass_blur = float((blur > 0.5 * train_median_blur).mean()) if N else 0.0
    frame_stats = {
        "n_kept": int(N),
        "n_train": int((split == SPLIT_TRAIN).sum()),
        "n_val": int((split == SPLIT_VAL).sum()),
        "n_test": int((split == SPLIT_TEST).sum()),
        "blur_pass_frac": pass_blur,
        "color_gain_warn_frac": float((gain > 0.05).mean()) if N else 0.0,
    }

    # ---- color ----
    color_stats = {
        "gain_mean": float(gain.mean()) if N else 0.0,
        "gain_p95": float(np.percentile(gain, 95)) if N else 0.0,
    }

    report = {
        "subject_id": subject_id,
        "n_channels_supported": int(n_supported),
        "n_channels_partial": int(n_partial),
        "n_channels_unsupported": int(n_unsupported),
        "critical_channels_ok": bool(critical_ok),
        "channels": ch_rows,
        "groups": group_rows,
        "pose": pose_rows,
        "pose_occupancy_yaw_pitch": pose_occupancy,
        "gaze_p99_rad": gaze_p99,
        "combinations": pairs[:200],
        "frames": frame_stats,
        "color": color_stats,
    }
    return report


# ---------------------------------------------------------------------------
# G0 gate (§10.6)
# ---------------------------------------------------------------------------
def gate_g0(report: dict, cfg: DictConfig) -> tuple[bool, list[str]]:
    f: list[str] = []
    fr = report["frames"]

    if fr["n_train"] + fr["n_val"] + fr["n_test"] == 0:
        f.append("no kept frames")
    # tracker validity: expressed as kept/total_valid -- we approximate by
    # checking that the *tracker* pass reported >= 97% valid; caller passes that in.
    # We check the coverage-side ones here:
    if report["n_channels_supported"] < 36:
        f.append(f"only {report['n_channels_supported']} supported channels (< 36)")
    if not report["critical_channels_ok"]:
        f.append("one or more critical channels not supported")

    # yaw / pitch per-side P95
    yaw = report["pose"]["yaw"]
    pitch = report["pose"]["pitch"]
    if yaw["p95_pos"] < 20.0 or yaw["p95_neg"] < 20.0:
        f.append(f"yaw P95 too narrow: +{yaw['p95_pos']:.1f}/-{yaw['p95_neg']:.1f} (<20)")
    if pitch["p95_pos"] < 10.0 or pitch["p95_neg"] < 10.0:
        f.append(f"pitch P95 too narrow: +{pitch['p95_pos']:.1f}/-{pitch['p95_neg']:.1f} (<10)")

    if fr["blur_pass_frac"] < 0.85:
        f.append(f"only {fr['blur_pass_frac']*100:.1f}% frames pass blur filter (<85%)")
    if fr["color_gain_warn_frac"] > 0.05:
        f.append(f"{fr['color_gain_warn_frac']*100:.1f}% frames have color gain > 5% (>5%)")

    return len(f) == 0, f


def tracker_validity_ok(tracker_valid_ratio: float) -> tuple[bool, str | None]:
    if tracker_valid_ratio < 0.97:
        return False, f"tracker valid ratio {tracker_valid_ratio*100:.1f}% < 97%"
    return True, None


# ---------------------------------------------------------------------------
# HTML (optional; helpful when reading the report by hand)
# ---------------------------------------------------------------------------
def write_html(report: dict, path: str | Path) -> None:
    rows = []
    for r in report["channels"]:
        cls = {"supported": "#c8f7c5", "partial": "#fde8b3", "unsupported": "#f7c5c5"}[r["status"]]
        rows.append(f"<tr style='background:{cls}'>"
                    f"<td>{r['name']}</td><td>{r['frames_above_0.25']}</td>"
                    f"<td>{r['p95']:.3f}</td><td>{r['status']}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>GALNP-S coverage {report['subject_id']}</title>
<style>body{{font-family:monospace;font-size:13px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:2px 6px}}</style>
</head><body>
<h2>Coverage report — {report['subject_id']}</h2>
<p><b>supported:</b> {report['n_channels_supported']} &nbsp;
<b>partial:</b> {report['n_channels_partial']} &nbsp;
<b>unsupported:</b> {report['n_channels_unsupported']} &nbsp;
<b>critical ok:</b> {report['critical_channels_ok']}</p>
<p><b>blur pass:</b> {report['frames']['blur_pass_frac']*100:.1f}% &nbsp;
<b>color warn:</b> {report['frames']['color_gain_warn_frac']*100:.2f}% &nbsp;
<b>kept:</b> {report['frames']['n_kept']}</p>
<h3>Channels</h3>
<table><tr><th>name</th><th>frames&gt;0.25</th><th>p95</th><th>status</th></tr>
{''.join(rows)}</table>
</body></html>"""
    Path(path).write_text(html)


def report_and_gate(frames_npz: str | Path, subject_id: str,
                    cfg: DictConfig, tracker_valid_ratio: float,
                    out_dir: str | Path) -> tuple[dict, bool, list[str]]:
    report = compute_coverage(frames_npz, subject_id)
    tv_ok, tv_msg = tracker_validity_ok(tracker_valid_ratio)
    ok, fails = gate_g0(report, cfg)
    if not tv_ok and tv_msg:
        fails = [tv_msg] + fails
        ok = False

    out_dir = Path(out_dir)
    save_json(report, out_dir / "coverage.json")
    write_html(report, out_dir / "coverage.html")

    if ok:
        log.info("G0 PASS")
    else:
        log.warning("G0 FAIL:")
        for m in fails:
            log.warning("  - %s", m)
    return report, ok, fails
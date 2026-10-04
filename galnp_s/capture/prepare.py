"""S0 capture preparation (§4.5, §5.1, §10.1, §10.2).

Runs on CPU. Consumes:
  - the raw session video
  - the tracker cache from `scripts/s0_track.py` (tracker.npz + channels.json)
  - the per-session time ranges from `subject.yaml`

Produces in `<workdir>/session<id>/`:
  - plate.npy               empty-background plate (§5.1)
  - headcrops.npy           (N, 512, 512, 3) uint8 RGB, memory-mappable
  - nativecrops.npy         (Nn, S, S, 3) uint8 RGB, S = session-fixed
  - frames.npz              per-frame metadata (see _save_metadata)
  - session.json            summary + config snapshot

Split ids in frames.npz["split"]:
  0 = val   1 = test   2 = train   3 = guard
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from galnp_s.utils.io import ensure_dir, save_json
from galnp_s.utils.logging import get_logger

log = get_logger(__name__)

# --- MediaPipe 478-landmark indices we rely on -------------------------------
LM_FOREHEAD_TOP = 10
LM_CHIN = 152
LM_EYE_A_OUTER = 33
LM_EYE_A_INNER = 133
LM_EYE_B_INNER = 362
LM_EYE_B_OUTER = 263
LM_IRIS_A = 468           # iris center, eye A
LM_IRIS_B = 473           # iris center, eye B

# --- constants ---------------------------------------------------------------
DEFAULT_FACE_OVAL_FACTOR = 1.6
DEFAULT_CROP_ANCHOR_X = 0.5
DEFAULT_CROP_ANCHOR_Y = 0.42

SPLIT_VAL, SPLIT_TEST, SPLIT_TRAIN, SPLIT_GUARD = 0, 1, 2, 3

NATIVE_MIN, NATIVE_MAX, NATIVE_STEP = 512, 1152, 32

# ---------------------------------------------------------------------------
# Photometric helpers (§5.1)
# ---------------------------------------------------------------------------
import os
import tempfile

def build_plate(video_path: str | Path,
                plate_ranges: Iterable[tuple[float, float]],
                fps: float,
                subsample: int = 10,
                row_chunk: int = 128,
                tmp_dir: str | Path | None = None) -> np.ndarray:
    """Median of the empty-scene recording, RGB uint8 (H, W, 3).

    Streams the video once (no seek-per-frame). Frames are written into a
    disk-backed memmap so peak RAM stays small even at 4K. The median is
    computed in row chunks, each chunk ~`row_chunk` rows.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    ranges = [(int(round(s * fps)), int(round(e * fps))) for s, e in plate_ranges]
    wanted: set[int] = set()
    for s, e in ranges:
        wanted.update(range(s, e, subsample))
    if not wanted:
        cap.release()
        raise RuntimeError("no plate frames requested")
    wanted = sorted(wanted)
    n_want = len(wanted)

    tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stack_path = tmp_dir / f"galnp_s_plate_stack_{os.getpid()}.npy"

    frames = None            # memmap, shape (n_want, H, W, 3) uint8
    h = w = None
    n_read = 0
    frame_idx = 0
    next_i = 0
    try:
        while next_i < n_want:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_idx == wanted[next_i]:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frames is None:
                    h, w = rgb.shape[:2]
                    frames = np.lib.format.open_memmap(
                        stack_path, mode="w+",
                        dtype=np.uint8, shape=(n_want, h, w, 3),
                    )
                    log.info("plate stack: %d frames at %dx%d -> %s",
                             n_want, w, h, stack_path)
                frames[next_i] = rgb
                n_read += 1
                next_i += 1
            frame_idx += 1
    finally:
        cap.release()

    if frames is None or n_read == 0:
        if stack_path.exists():
            stack_path.unlink()
        raise RuntimeError("no plate frames read")

    # Chunked median over rows. Each chunk is (n_read, row_chunk, W, 3) uint8.
    med = np.empty((h, w, 3), dtype=np.uint8)
    step = max(1, int(row_chunk))
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        med[y0:y1] = np.median(frames[:n_read, y0:y1], axis=0).astype(np.uint8)

    del frames
    if stack_path.exists():
        stack_path.unlink()

    log.info("plate ready from %d frames", n_read)
    return med
    
def compute_matte(frame_rgb: np.ndarray, plate_rgb: np.ndarray,
                  tau_lo: float = 6.0, tau_hi: float = 14.0) -> np.ndarray:
    """Soft foreground matte in [0,1] from Lab distance to the plate (§5.1)."""
    f = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    p = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(f - p, axis=2)
    m = np.clip((d - tau_lo) / max(tau_hi - tau_lo, 1e-6), 0.0, 1.0)
    m8 = cv2.medianBlur((m * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    return m8


def fit_color_transform(frame_rgb: np.ndarray, plate_rgb: np.ndarray,
                        matte: np.ndarray, dilate_px: int = 5) -> np.ndarray:
    """Least-squares (4,3) affine fit on background pixels (§5.1).

    out = [R,G,B,1] @ W.  Fit so out ~= plate on background.
    """
    fg = (matte > 0.1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    fg_dil = cv2.dilate(fg, k)
    bg = fg_dil == 0
    n_bg = int(bg.sum())
    if n_bg < 200:
        W = np.zeros((4, 3), dtype=np.float64)
        W[:3] = np.eye(3)
        return W
    A = np.concatenate(
        [frame_rgb[bg].astype(np.float64), np.ones((n_bg, 1), dtype=np.float64)], axis=1
    )
    B = plate_rgb[bg].astype(np.float64)
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    return W.astype(np.float64)


def apply_color_transform(frame_rgb: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply (4,3) affine; returns uint8 RGB."""
    h, w = frame_rgb.shape[:2]
    A = np.concatenate(
        [frame_rgb.reshape(-1, 3).astype(np.float64), np.ones((h * w, 1), dtype=np.float64)],
        axis=1,
    )
    out = (A @ W).reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_gain(W: np.ndarray) -> float:
    """||M - I||_F, for the color QA warning (§5.1)."""
    return float(np.linalg.norm(W[:3] - np.eye(3)))


# ---------------------------------------------------------------------------
# Geometry helpers (§4.5)
# ---------------------------------------------------------------------------
def face_bbox(lm_px: np.ndarray, w: int, h: int, margin: float = 0.15) -> tuple[int, int, int, int]:
    xy = lm_px[:, :2]
    x_min, y_min = xy.min(0)
    x_max, y_max = xy.max(0)
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin
    x1 = max(0, int(x_min - dx))
    y1 = max(0, int(y_min - dy))
    x2 = min(w, int(x_max + dx))
    y2 = min(h, int(y_max + dy))
    return x1, y1, x2, y2


def blur_score(frame_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_height_px(lm_px: np.ndarray) -> float:
    """Rotating head does not change this because we use the distance between
    landmark 10 (forehead top) and 152 (chin) — a 3D-ish measure."""
    return float(np.linalg.norm(lm_px[LM_FOREHEAD_TOP, :2] - lm_px[LM_CHIN, :2]))


def eye_centers(lm_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cA = (lm_px[LM_EYE_A_OUTER, :2] + lm_px[LM_EYE_A_INNER, :2]) / 2.0
    cB = (lm_px[LM_EYE_B_OUTER, :2] + lm_px[LM_EYE_B_INNER, :2]) / 2.0
    return cA, cB


def head_crop_affine(lm_px: np.ndarray, dest_side: int,
                     factor: float = DEFAULT_FACE_OVAL_FACTOR,
                     ax: float = DEFAULT_CROP_ANCHOR_X,
                     ay: float = DEFAULT_CROP_ANCHOR_Y) -> np.ndarray:
    """Return a 2x3 similarity that maps the source to a `dest_side` square
    where the eye midpoint sits at (ax, ay) and the face-oval height maps to
    dest_side / factor."""
    cA, cB = eye_centers(lm_px)
    M = (cA + cB) * 0.5
    theta = float(np.arctan2(cB[1] - cA[1], cB[0] - cA[0]))
    fh = face_height_px(lm_px)
    if fh < 1.0:
        raise ValueError(f"degenerate face height: {fh}")
    s = dest_side / (factor * fh)
    c, si = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -si], [si, c]], dtype=np.float64)
    anchor = np.array([ax * dest_side, ay * dest_side], dtype=np.float64)
    t = anchor - s * (R @ M)
    A = np.zeros((2, 3), dtype=np.float32)
    A[:2, :2] = (s * R).astype(np.float32)
    A[:, 2] = t.astype(np.float32)
    return A


def warp_crop(frame_rgb: np.ndarray, affine: np.ndarray, dest_side: int) -> np.ndarray:
    return cv2.warpAffine(
        frame_rgb, affine, (dest_side, dest_side),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def euler_from_transform(T: np.ndarray) -> np.ndarray:
    """YPR Euler (deg) from MediaPipe's 4x4 facial transform. S1 will replace
    this with a proper Procrustes fit against the canonical mesh."""
    R = T[:3, :3].astype(np.float64)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = U @ Vt
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([yaw, pitch, roll]).astype(np.float32)


def gaze_radians(lm_px: np.ndarray) -> np.ndarray:
    """Per-eye (yaw, pitch) in radians: [A_yaw, A_pitch, B_yaw, B_pitch]."""
    out: list[float] = []
    for outer, inner, iris in ((LM_EYE_A_OUTER, LM_EYE_A_INNER, LM_IRIS_A),
                               (LM_EYE_B_OUTER, LM_EYE_B_INNER, LM_IRIS_B)):
        corners = lm_px[[outer, inner], :2]
        iris_pt = lm_px[iris, :2]
        center = corners.mean(0)
        axis = corners[1] - corners[0]
        half_w = float(np.linalg.norm(axis)) * 0.5
        if half_w < 1e-3:
            out.extend([0.0, 0.0])
            continue
        u = axis / (2.0 * half_w)
        v = np.array([-u[1], u[0]])
        dx = float(np.dot(iris_pt - center, u)) / half_w
        dy = float(np.dot(iris_pt - center, v)) / half_w
        out.extend([float(np.arctan(dx)), float(np.arctan(dy))])
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Splits (§10.2)
# ---------------------------------------------------------------------------
def pick_blocks(f_start: int, f_end: int, e: np.ndarray, valid: np.ndarray,
                fps: float, block_s: float, n_pick: int, guard_s: float,
                min_valid_frac: float = 0.8) -> list[int]:
    """Pick `n_pick` non-overlapping `block_s`-second windows from
    [f_start, f_end), closest first to the segment's median e-statistic."""
    win = int(round(block_s * fps))
    guard = int(round(guard_s * fps))
    if f_end - f_start < win:
        raise ValueError(f"segment too short: {f_end - f_start} < {win}")
    starts = np.arange(f_start, f_end - win + 1, dtype=np.int64)
    summaries: list[np.ndarray] = []
    kept_starts: list[int] = []
    for s in starts:
        sl = slice(int(s), int(s) + win)
        v = valid[sl]
        if v.sum() < min_valid_frac * win:
            continue
        summaries.append(e[sl][v].mean(0))
        kept_starts.append(int(s))
    if len(kept_starts) < n_pick:
        raise ValueError(f"only {len(kept_starts)} valid windows, need {n_pick}")
    S = np.stack(summaries, 0)
    med = np.median(S, axis=0)
    d = np.linalg.norm(S - med, axis=1)
    order = np.argsort(d)
    picked: list[int] = []
    for _ in range(n_pick):
        for k in order:
            s = kept_starts[int(k)]
            if any(abs(s - p) < win + guard for p in picked):
                continue
            picked.append(s)
            break
        else:
            raise ValueError("could not pick enough non-overlapping windows")
    return sorted(picked)


def compute_splits(segments: dict, e: np.ndarray, valid: np.ndarray,
                   fps: float, cfg: DictConfig) -> np.ndarray:
    """Return int8 array (T,) with SPLIT_{VAL,TEST,TRAIN,GUARD}."""
    T = len(e)
    splits = np.full(T, SPLIT_TRAIN, dtype=np.int8)
    block_s = float(cfg.splits.block_seconds)
    guard_s = float(cfg.splits.guard_seconds)
    guard = int(round(guard_s * fps))
    win = int(round(block_s * fps))

    # (segment key, assign first-window to val / second to test)
    triple = ("speech", "emotive", "pose_sweep")
    for key in triple:
        if key not in segments:
            raise KeyError(f"missing segment '{key}' in subject config")
        s_s, e_s = segments[key]
        f0 = int(round(s_s * fps))
        f1 = int(round(e_s * fps))
        starts = pick_blocks(f0, f1, e, valid, fps, block_s, 2, guard_s)
        for i, ws in enumerate(starts):
            assign = SPLIT_VAL if i == 0 else SPLIT_TEST
            we = ws + win
            splits[ws:we] = assign
            for tt in range(max(0, ws - guard), ws):
                if splits[tt] == SPLIT_TRAIN:
                    splits[tt] = SPLIT_GUARD
            for tt in range(we, min(T, we + guard)):
                if splits[tt] == SPLIT_TRAIN:
                    splits[tt] = SPLIT_GUARD
    return splits


def compute_motion(e: np.ndarray) -> np.ndarray:
    m = np.zeros(len(e), dtype=np.float32)
    if len(e) > 1:
        m[1:] = np.abs(np.diff(e, axis=0)).sum(axis=1)
    return m


# ---------------------------------------------------------------------------
# Native-crop pool (§8.6 relies on native refs at inference)
# ---------------------------------------------------------------------------
def _fps(feats: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    N = len(feats)
    n = min(n, N)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    sel = np.zeros(n, dtype=np.int64)
    sel[0] = seed
    d = np.linalg.norm(feats - feats[seed], axis=1)
    for i in range(1, n):
        idx = int(np.argmax(d))
        sel[i] = idx
        d = np.minimum(d, np.linalg.norm(feats - feats[idx], axis=1))
    return sel


def choose_native_pool(train_idx: np.ndarray, e: np.ndarray, pose: np.ndarray,
                       sharpness: np.ndarray, k: int) -> np.ndarray:
    """FPS over (e, pose/6) on training frames; returns row indices into
    train_idx (i.e. absolute frame indices)."""
    if len(train_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    feats = np.concatenate([e[train_idx], pose[train_idx] / 6.0], axis=1).astype(np.float32)
    # Prefer high-sharpness frames as FPS seed.
    seed = int(np.argmax(sharpness[train_idx])) if len(train_idx) else 0
    picked = _fps(feats, k, seed=seed)
    return train_idx[picked]


# ---------------------------------------------------------------------------
# Main session pass
# ---------------------------------------------------------------------------
def _round_native(x: float) -> int:
    r = int(round(x / NATIVE_STEP) * NATIVE_STEP)
    return max(NATIVE_MIN, min(NATIVE_MAX, r))


def process_session(video_path: str | Path,
                    tracker_npz: str | Path,
                    out_dir: str | Path,
                    session_id: int,
                    session_cfg: DictConfig,
                    cfg: DictConfig) -> dict:
    out_dir = ensure_dir(out_dir)
    t_start = time.time()

    # 1. tracker
    tr = np.load(tracker_npz)
    lm_px = tr["landmarks_px"]           # (T, 478, 3)
    e_all = tr["blendshapes"].astype(np.float32)   # (T, 52)
    T_all = tr["transform"]              # (T, 4, 4)
    valid_all = tr["valid"].astype(bool) # (T,)
    T = int(len(valid_all))
    log.info("session %d: %d tracked frames, %d valid (%.1f%%)",
             session_id, T, int(valid_all.sum()), 100.0 * valid_all.mean())

    fps = float(session_cfg.fps)

    # 2. face height per frame (from landmarks)
    fh = np.zeros(T, dtype=np.float32)
    for t in np.where(valid_all)[0]:
        fh[t] = face_height_px(lm_px[t])

    # 3. splits
    splits = compute_splits(dict(session_cfg.segments), e_all, valid_all, fps, cfg)
    n_val = int((splits == SPLIT_VAL).sum())
    n_test = int((splits == SPLIT_TEST).sum())
    n_train = int((splits == SPLIT_TRAIN).sum())
    log.info("splits: val=%d test=%d train=%d guard=%d",
             n_val, n_test, n_train, int((splits == SPLIT_GUARD).sum()))

    # 4. native side from training frames
    train_mask = (splits == SPLIT_TRAIN) & valid_all & (fh > 0)
    if train_mask.sum() == 0:
        raise RuntimeError("no valid training frames")
    med_fh = float(np.median(fh[train_mask]))
    native_side = _round_native(cfg.crop.side_factor * med_fh)
    log.info("median training face height %.1f px -> native_side %d", med_fh, native_side)

    # 5. motion + thinning
    motion = compute_motion(e_all)
    p85 = float(np.percentile(motion[train_mask], 85))
    even = (np.arange(T) % 2) == 0
    keep_train = train_mask & (even | (motion > p85))
    kept = (keep_train | (splits == SPLIT_VAL) | (splits == SPLIT_TEST)) & valid_all
    kept_idx = np.where(kept)[0]
    N = int(len(kept_idx))
    log.info("kept %d frames (train=%d val=%d test=%d)",
             N, int((splits[kept_idx] == SPLIT_TRAIN).sum()),
             int((splits[kept_idx] == SPLIT_VAL).sum()),
             int((splits[kept_idx] == SPLIT_TEST).sum()))

    # 6. plate
    plate_ranges = [session_cfg.segments[f"plate_{w}"]
                    for w in ("start", "end") if f"plate_{w}" in session_cfg.segments]
    if not plate_ranges:
        raise KeyError("session config must define 'plate_start' (and ideally 'plate_end')")
    plate = build_plate(video_path, plate_ranges, fps, subsample=10)
    np.save(out_dir / "plate.npy", plate)
    log.info("plate saved (%s)", plate.shape)

    # 7. choose native pool
    train_kept = kept_idx[splits[kept_idx] == SPLIT_TRAIN]
    if bool(cfg.capture.get("store_native_all", False)):
        native_pool = kept_idx
        native_row_of_frame = -np.ones(T, dtype=np.int64)
        for i, f in enumerate(native_pool):
            native_row_of_frame[f] = i
    else:
        k_native = int(cfg.capture.get("native_pool_size", 1024))
        # run FPS on training frames (sharpness not known yet -> use placeholder)
        # We use valid fractions instead of sharpness here; sharper filtering is
        # applied later in bank_select.
        pool = choose_native_pool(train_kept, e_all, np.zeros((T, 3), np.float32),
                                  np.zeros(T, np.float32), k_native)
        native_pool = np.unique(np.concatenate([pool,
                                                kept_idx[splits[kept_idx] == SPLIT_VAL],
                                                kept_idx[splits[kept_idx] == SPLIT_TEST]]))
        native_pool = np.sort(native_pool)
        native_row_of_frame = -np.ones(T, dtype=np.int64)
        for i, f in enumerate(native_pool):
            native_row_of_frame[f] = i
    Nn = int(len(native_pool))
    log.info("native pool: %d frames at %d^2", Nn, native_side)

    # 8. memmaps
    head_path = out_dir / "headcrops.npy"
    nat_path = out_dir / "nativecrops.npy"
    if head_path.exists():
        head_path.unlink()
    if nat_path.exists():
        nat_path.unlink()
    headcrops = np.lib.format.open_memmap(head_path, mode="w+", dtype=np.uint8,
                                          shape=(N, 512, 512, 3))
    nativecrops = np.lib.format.open_memmap(nat_path, mode="w+", dtype=np.uint8,
                                            shape=(Nn, native_side, native_side, 3))

    # 9. metadata buffers
    blur_a = np.zeros(N, np.float32)
    gain_a = np.zeros(N, np.float32)
    motion_a = np.zeros(N, np.float32)
    e_a = np.zeros((N, 52), np.float32)
    pose_a = np.zeros((N, 3), np.float32)
    gaze_a = np.zeros((N, 4), np.float32)
    A512_a = np.zeros((N, 2, 3), np.float32)
    Anat_a = np.zeros((N, 2, 3), np.float32)
    split_a = np.zeros(N, np.int8)

    # 10. per-frame loop
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video and n_video != T:
        log.warning("video frames (%d) != tracker frames (%d)", n_video, T)

    kept_set = np.zeros(T, dtype=bool)
    kept_set[kept_idx] = True

    H = W = None
    write_i = 0
    t0 = time.time()
    pbar = tqdm(total=N, desc=f"S0 sess{session_id}", unit="f")
    try:
        for t in range(T):
            ok, bgr = cap.read()
            if not ok:
                log.warning("video ended at frame %d/%d", t, T)
                break
            if not kept_set[t]:
                continue
            if not valid_all[t]:
                continue

            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if H is None:
                H, W = frame_rgb.shape[:2]

            matte = compute_matte(frame_rgb, plate)
            Wm = fit_color_transform(frame_rgb, plate, matte)
            gain = transform_gain(Wm)
            frame_norm = apply_color_transform(frame_rgb, Wm)

            lm = lm_px[t]
            bbox = face_bbox(lm, W, H)
            blur = blur_score(frame_norm, bbox)

            A512 = head_crop_affine(lm, 512)
            crop512 = warp_crop(frame_norm, A512, 512)
            headcrops[write_i] = crop512

            if native_row_of_frame[t] >= 0:
                Anat = head_crop_affine(lm, native_side)
                nativecrops[native_row_of_frame[t]] = warp_crop(frame_norm, Anat, native_side)
                Anat_a[write_i] = Anat

            blur_a[write_i] = blur
            gain_a[write_i] = gain
            motion_a[write_i] = motion[t]
            e_a[write_i] = e_all[t]
            pose_a[write_i] = euler_from_transform(T_all[t])
            gaze_a[write_i] = gaze_radians(lm)
            A512_a[write_i] = A512
            split_a[write_i] = splits[t]
            write_i += 1
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    if write_i != N:
        log.warning("wrote %d/%d frames (video truncated?)", write_i, N)
        # Truncate metadata views
        headcrops = headcrops[:write_i]
        N = write_i

    headcrops.flush() if hasattr(headcrops, "flush") else None
    nativecrops.flush() if hasattr(nativecrops, "flush") else None

    # 11. save metadata
    np.savez_compressed(
        out_dir / "frames.npz",
        orig_idx=kept_idx[:N].astype(np.int64),
        session=np.full(N, int(session_id), dtype=np.int8),
        blur=blur_a[:N],
        color_gain=gain_a[:N],
        motion=motion_a[:N],
        e=e_a[:N],
        pose_deg=pose_a[:N],
        gaze_rad=gaze_a[:N],
        crop_affine_512=A512_a[:N],
        crop_affine_nat=Anat_a[:N],
        split=split_a[:N],
        native_row=native_row_of_frame[kept_idx[:N]].astype(np.int64),
        native_side=np.int32(native_side),
    )

    # 12. summary
    summary = {
        "session_id": int(session_id),
        "video": str(video_path),
        "n_frames_video": int(T),
        "n_kept": int(N),
        "n_valid": int(valid_all.sum()),
        "native_side": int(native_side),
        "native_pool_size": int(len(native_pool)),
        "fps": float(fps),
        "splits": {
            "val": int((split_a[:N] == SPLIT_VAL).sum()),
            "test": int((split_a[:N] == SPLIT_TEST).sum()),
            "train": int((split_a[:N] == SPLIT_TRAIN).sum()),
        },
        "p85_motion_train": float(p85),
        "median_face_height_px": float(med_fh),
        "elapsed_s": float(time.time() - t_start),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(summary, out_dir / "session.json")
    log.info("session %d done in %.1f s -> %s", session_id, summary["elapsed_s"], out_dir)
    return summary
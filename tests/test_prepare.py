import numpy as np

from galnp_s.capture.prepare import (
    build_plate, compute_matte, fit_color_transform, apply_color_transform,
    face_height_px, eye_centers, head_crop_affine, warp_crop,
    euler_from_transform, gaze_radians, compute_motion,
    compute_splits, SPLIT_VAL, SPLIT_TEST, SPLIT_TRAIN, SPLIT_GUARD,
    _fps,
)


def _dummy_landmarks(h=720, w=1280, scale=1.0, cx=None, cy=None):
    """Synthetic 478 landmarks: face features at plausible positions."""
    lm = np.zeros((478, 3), dtype=np.float32)
    cx = cx if cx is not None else w / 2
    cy = cy if cy is not None else h / 2
    # arbitrary but stable positions for our tested indices
    lm[10] = [cx, cy - 90 * scale, 0]       # forehead
    lm[152] = [cx, cy + 90 * scale, 0]      # chin
    lm[33] = [cx - 60 * scale, cy - 30 * scale, 0]   # eye A outer
    lm[133] = [cx - 20 * scale, cy - 30 * scale, 0]  # eye A inner
    lm[362] = [cx + 20 * scale, cy - 30 * scale, 0]  # eye B inner
    lm[263] = [cx + 60 * scale, cy - 30 * scale, 0]  # eye B outer
    lm[468] = [cx - 40 * scale, cy - 30 * scale, 0]  # iris A
    lm[473] = [cx + 40 * scale, cy - 30 * scale, 0]  # iris B
    return lm


def test_face_height_px():
    lm = _dummy_landmarks(scale=1.0)
    assert face_height_px(lm) == 180.0


def test_eye_centers_symmetric():
    lm = _dummy_landmarks()
    cA, cB = eye_centers(lm)
    assert np.allclose(cA[0] + cB[0], 2 * 640.0, atol=1e-3)


def test_head_crop_affine_maps_midpoint_to_anchor():
    lm = _dummy_landmarks(scale=1.0)
    A = head_crop_affine(lm, dest_side=512)
    cA, cB = eye_centers(lm)
    M = (cA + cB) * 0.5
    p = A[:2, :2] @ M + A[:, 2]
    # anchor x=256, y=0.42*512=215.04
    assert abs(p[0] - 256.0) < 1e-3
    assert abs(p[1] - 0.42 * 512) < 1e-3


def test_head_crop_affine_scale():
    lm = _dummy_landmarks(scale=1.0)
    A = head_crop_affine(lm, dest_side=512, factor=1.6)
    # face height 180 -> should map to 512/1.6 = 320 px
    top = A[:2, :2] @ lm[10, :2] + A[:, 2]
    bot = A[:2, :2] @ lm[152, :2] + A[:, 2]
    assert abs(np.linalg.norm(top - bot) - 320.0) < 1.0


def test_head_crop_affine_rotates_to_horizontal():
    lm = _dummy_landmarks()
    # rotate eye line by 20 deg
    th = np.deg2rad(20.0)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    cx, cy = 640.0, 360.0
    for i in (33, 133, 362, 263):
        lm[i, :2] = R @ (lm[i, :2] - [cx, cy]) + [cx, cy]
    A = head_crop_affine(lm, dest_side=512)
    cA, cB = eye_centers(lm)
    pa = A[:2, :2] @ cA + A[:, 2]
    pb = A[:2, :2] @ cB + A[:, 2]
    assert abs(pa[1] - pb[1]) < 1e-3, "eyes should land at the same y in crop"


def test_warp_crop_shape():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    lm = _dummy_landmarks()
    A = head_crop_affine(lm, dest_side=512)
    out = warp_crop(img, A, 512)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8


def test_matte_zero_on_identical_frame():
    rng = np.random.default_rng(0)
    plate = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    matte = compute_matte(plate.copy(), plate)
    assert matte.max() < 1e-3


def test_matte_one_on_far_frame():
    rng = np.random.default_rng(0)
    plate = np.zeros((64, 64, 3), dtype=np.uint8)
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)
    matte = compute_matte(frame, plate)
    assert matte.mean() > 0.9


def test_color_transform_identity_on_matching():
    rng = np.random.default_rng(0)
    plate = (rng.random((64, 64, 3)) * 200 + 20).astype(np.uint8)
    matte = np.zeros((64, 64), dtype=np.float32)
    W = fit_color_transform(plate.copy(), plate, matte)
    assert np.allclose(W[:3], np.eye(3), atol=0.05)
    assert np.allclose(W[3], 0.0, atol=2.0)


def test_color_transform_shift():
    rng = np.random.default_rng(0)
    plate = np.full((64, 64, 3), 100, dtype=np.uint8)
    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    matte = np.zeros((64, 64), dtype=np.float32)
    W = fit_color_transform(frame, plate, matte)
    out = apply_color_transform(frame, W)
    assert np.abs(out.astype(int) - 100).max() <= 2


def test_euler_identity():
    T = np.eye(4, dtype=np.float32)
    ypr = euler_from_transform(T)
    assert np.allclose(ypr, 0.0, atol=1e-5)


def test_euler_yaw():
    # Rotation about Z (the "yaw" axis for our ZYX extraction).
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th),  np.cos(th), 0],
                  [0,           0,          1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    ypr = euler_from_transform(T)
    assert abs(ypr[0] - 30.0) < 1e-3
    assert abs(ypr[1]) < 1e-3
    assert abs(ypr[2]) < 1e-3


def test_euler_pitch():
    th = np.deg2rad(20.0)
    R = np.array([[ np.cos(th), 0, np.sin(th)],
                  [ 0,          1, 0         ],
                  [-np.sin(th), 0, np.cos(th)]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    ypr = euler_from_transform(T)
    assert abs(ypr[0]) < 1e-3
    assert abs(ypr[1] - 20.0) < 1e-3
    assert abs(ypr[2]) < 1e-3


def test_euler_roll():
    th = np.deg2rad(15.0)
    R = np.array([[1, 0,           0          ],
                  [0, np.cos(th), -np.sin(th)],
                  [0, np.sin(th),  np.cos(th)]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    ypr = euler_from_transform(T)
    assert abs(ypr[0]) < 1e-3
    assert abs(ypr[1]) < 1e-3
    assert abs(ypr[2] - 15.0) < 1e-3

def test_gaze_shape():
    lm = _dummy_landmarks()
    g = gaze_radians(lm)
    assert g.shape == (4,) and g.dtype == np.float32


def test_gaze_centered():
    lm = _dummy_landmarks()
    g = gaze_radians(lm)
    assert np.allclose(g, 0.0, atol=1e-3)


def test_compute_motion():
    e = np.zeros((5, 52), dtype=np.float32)
    e[2] = 1.0
    m = compute_motion(e)
    assert m[0] == 0.0 and m[2] == 52.0 and m[4] == 0.0


def test_compute_splits_basic():
    from types import SimpleNamespace
    T = 30 * 300  # 300 s
    e = np.zeros((T, 52), dtype=np.float32)
    valid = np.ones(T, dtype=bool)
    segments = {
        "speech": (60.0, 180.0),
        "emotive": (180.0, 240.0),
        "pose_sweep": (240.0, 290.0),
    }
    cfg = SimpleNamespace(splits=SimpleNamespace(block_seconds=10.0, guard_seconds=1.0))
    splits = compute_splits(segments, e, valid, fps=30.0, cfg=cfg)
    assert (splits == SPLIT_VAL).sum() == 3 * 300
    assert (splits == SPLIT_TEST).sum() == 3 * 300
    # guard should be present
    assert (splits == SPLIT_GUARD).sum() > 0
    # train should be the rest
    assert (splits == SPLIT_TRAIN).sum() + 6 * 300 + (splits == SPLIT_GUARD).sum() == T


def test_fps_returns_distinct_indices():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((100, 5)).astype(np.float32)
    idx = _fps(feats, 10, seed=0)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10
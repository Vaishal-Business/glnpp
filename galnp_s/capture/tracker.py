"""Frozen MediaPipe Face Landmarker wrapper (§4.1).

Outputs per frame, at native video resolution:
  landmarks_norm : (478, 3) float32 -- normalized (x, y, z) in [0,1] for x,y;
                    z is roughly face-scale-relative depth.
  landmarks_px   : (478, 3) float32 -- x, y in pixels, z in pixels scaled by
                    image width (convenient for IOD normalization).
  blendshapes    : (52,)    float32 in [0, 1] (order = channels.CHANNEL_NAMES)
  transform      : (4, 4)   float32 -- facial transformation matrix.
  valid          : bool

Run with `RunningMode.VIDEO` so MediaPipe can use temporal smoothing.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from galnp_s.capture import channels as C
from galnp_s.utils.io import ensure_dir
from galnp_s.utils.logging import get_logger

log = get_logger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(model_path: str | Path) -> Path:
    """Download the .task file once and cache it."""
    p = Path(model_path)
    if p.exists():
        return p
    ensure_dir(p.parent)
    log.info("downloading MediaPipe face_landmarker.task -> %s", p)
    urllib.request.urlretrieve(_MODEL_URL, p)
    return p


@dataclass
class FrameTrack:
    landmarks_norm: np.ndarray   # (478, 3) f32
    landmarks_px: np.ndarray     # (478, 3) f32
    blendshapes: np.ndarray      # (52,)   f32
    transform: np.ndarray        # (4, 4)  f32
    valid: bool


class FaceTracker:
    """Thin, deterministic wrapper. One instance per video stream."""

    def __init__(self, model_path: str | Path, min_face_detection_confidence: float = 0.5):
        # Import here so that `import galnp_s.capture.tracker` fails loudly only when used.
        import mediapipe as mp

        self._mp = mp
        model_path = ensure_model(model_path)
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_face_detection_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        # Record the actual blendshape name order this build emits, for channels.json.
        self.emitted_names: list[str] | None = None

    # ----------------------------------------------------------------------
    def _order_blendshapes(self, categories) -> np.ndarray:
        """Map this build's Category list to our fixed 52-d order."""
        by_name = {c.category_name: float(c.score) for c in categories}
        if self.emitted_names is None:
            self.emitted_names = [c.category_name for c in categories]
        v = np.zeros(len(C.CHANNEL_NAMES), dtype=np.float32)
        for i, name in enumerate(C.CHANNEL_NAMES):
            v[i] = by_name.get(name, 0.0)
        return v

    # ----------------------------------------------------------------------
    def track_video(self, video_path: str | Path) -> Iterator[FrameTrack]:
        """Yield one FrameTrack per decoded frame. Frames where no face is
        detected are yielded with valid=False and NaN-filled arrays (the caller
        decides how to handle them; §F4 = hold-last, then blend to neutral)."""
        mp = self._mp
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(idx * 1000.0 / fps))
                res = self._landmarker.detect_for_video(mp_img, ts_ms)

                if not res.face_landmarks:
                    nan3 = np.full((478, 3), np.nan, dtype=np.float32)
                    yield FrameTrack(nan3, nan3.copy(),
                                     np.zeros(52, dtype=np.float32),
                                     np.eye(4, dtype=np.float32), False)
                    idx += 1
                    continue

                lms = res.face_landmarks[0]                     # 478 NormalizedLandmark
                ln = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                lp = ln.copy()
                lp[:, 0] *= w
                lp[:, 1] *= h
                lp[:, 2] *= w                                    # z-scale = width (matches MP convention)

                bs = self._order_blendshapes(res.face_blendshapes[0]) if res.face_blendshapes else np.zeros(52, np.float32)

                if res.facial_transformation_matrixes:
                    T = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float32)
                else:
                    T = np.eye(4, dtype=np.float32)

                yield FrameTrack(ln, lp, bs, T, True)
                idx += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._landmarker.close()

    # context-manager sugar
    def __enter__(self):  return self
    def __exit__(self, *a): self.close()


# --------------------------------------------------------------------------
def track_to_npz(video_path: str | Path,
                 out_npz: str | Path,
                 model_path: str | Path,
                 progress: bool = True) -> dict:
    """Run the tracker over a whole video, save an .npz plus a `channels.json`
    side-car recording the exact emitted names (§4.1, §F25)."""
    from tqdm import tqdm

    out_npz = Path(out_npz)
    ensure_dir(out_npz.parent)

    tracks: list[FrameTrack] = []
    tracker = FaceTracker(model_path)
    it: Iterable[FrameTrack] = tracker.track_video(video_path)
    if progress:
        it = tqdm(it, desc=f"track {Path(video_path).name}", unit="frame")
    for ft in it:
        tracks.append(ft)
    tracker.close()

    T = len(tracks)
    ln = np.stack([t.landmarks_norm for t in tracks])
    lp = np.stack([t.landmarks_px   for t in tracks])
    bs = np.stack([t.blendshapes    for t in tracks])
    tf = np.stack([t.transform      for t in tracks])
    ok = np.array([t.valid          for t in tracks], dtype=bool)

    np.savez_compressed(out_npz,
                        landmarks_norm=ln, landmarks_px=lp,
                        blendshapes=bs, transform=tf, valid=ok)

    meta = {
        "video": str(video_path),
        "n_frames": T,
        "channel_names": list(C.CHANNEL_NAMES),
        "emitted_names": tracker.emitted_names,
        "n_valid": int(ok.sum()),
        "valid_ratio": float(ok.mean()) if T else 0.0,
    }
    (out_npz.parent / "channels.json").write_text(json.dumps(meta, indent=2))
    log.info("tracked %d frames (%d valid) -> %s", T, int(ok.sum()), out_npz)
    return meta
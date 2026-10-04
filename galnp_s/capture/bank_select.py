"""Reference-bank selection (§5.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from galnp_s.capture import channels as C
from galnp_s.capture.prepare import SPLIT_TRAIN
from galnp_s.utils.io import save_json
from galnp_s.utils.logging import get_logger

log = get_logger(__name__)


def _fps_from_seed(feats: np.ndarray, n: int, seed_feats: np.ndarray) -> np.ndarray:
    """Farthest-point sampling over `feats`, seeded by `seed_feats`."""
    N = len(feats)
    n = min(n, N)
    if n <= 0 or N == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.min(np.linalg.norm(feats[:, None, :] - seed_feats[None, :, :], axis=2), axis=1)
    picked = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = int(np.argmax(d))
        picked[i] = idx
        d = np.minimum(d, np.linalg.norm(feats - feats[idx], axis=1))
    return picked


def _weights_from_train(e_train: np.ndarray, supported_mask: np.ndarray) -> np.ndarray:
    sigma = e_train.std(axis=0)
    w = 1.0 / np.maximum(sigma, 0.05)
    # Channel-specific boosts (§5.2)
    for ch in ("eyeBlinkLeft", "eyeBlinkRight"):
        if ch in C.NAME_TO_IDX:
            w[C.NAME_TO_IDX[ch]] *= 2.0
    if "jawOpen" in C.NAME_TO_IDX:
        w[C.NAME_TO_IDX["jawOpen"]] *= 1.5
    # `_neutral` is a placeholder; it must never contribute to retrieval distances.
    if "_neutral" in C.NAME_TO_IDX:
        w[C.NAME_TO_IDX["_neutral"]] = 0.0
    w[~supported_mask] = 0.0
    return w

def _neighbour_reject(feats: np.ndarray, k: int = 3, factor: float = 3.0) -> np.ndarray:
    """Return mask of frames with >= k neighbours within factor * median-kNN."""
    N = len(feats)
    if N < k + 1:
        return np.ones(N, dtype=bool)
    # subsample for speed if large
    if N > 4000:
        idx = np.linspace(0, N - 1, 4000).astype(np.int64)
        sample = feats[idx]
    else:
        sample = feats
    # pairwise distances (chunked)
    chunk = 512
    knn = np.empty(len(sample), dtype=np.float32)
    for i in range(0, len(sample), chunk):
        s = sample[i:i + chunk]
        d = np.linalg.norm(s[:, None, :] - sample[None, :, :], axis=2)
        d.sort(axis=1)
        knn[i:i + chunk] = d[:, k]
    med = float(np.median(knn))
    thresh = factor * med
    mask = np.zeros(N, dtype=bool)
    for i in range(0, N, chunk):
        s = feats[i:i + chunk]
        d = np.linalg.norm(s[:, None, :] - sample[None, :, :], axis=2)
        cnt = (d < thresh).sum(axis=1)
        mask[i:i + chunk] = cnt >= k
    return mask


# ---------------------------------------------------------------------------
def select_bank(frames_npz: str | Path,
                coverage_json: str | Path,
                cfg: DictConfig,
                out_path: str | Path) -> dict:
    d = np.load(frames_npz)
    e = d["e"].astype(np.float32)
    pose = d["pose_deg"].astype(np.float32)
    split = d["split"]
    blur = d["blur"]
    nat_row = d["native_row"].astype(np.int64)
    N = len(e)

    import json
    cov = json.loads(Path(coverage_json).read_text())
    supported_mask = np.array(
        [next((r["status"] == "supported" for r in cov["channels"] if r["name"] == name), False)
         for name in C.CHANNEL_NAMES],
        dtype=bool,
    )

    # 1. Candidates: training frames only, valid = present, plus sharpness filter,
    #    plus neighbour test.
    train_mask = (split == SPLIT_TRAIN)
    if train_mask.sum() < 100:
        raise RuntimeError(f"not enough training frames: {int(train_mask.sum())}")
    sharpness_ok = blur >= np.median(blur[train_mask])
    cand_mask = train_mask & sharpness_ok

    # weight vector
    w = _weights_from_train(e[train_mask], supported_mask)

    # joint feature: [w_U * e_U, w_L * e_L, pose/6 deg]
    u_idx = [C.NAME_TO_IDX[n] for n in C.UPPER_GROUP if n in C.NAME_TO_IDX]
    l_idx = [C.NAME_TO_IDX[n] for n in C.LOWER_GROUP if n in C.NAME_TO_IDX]
    feats = np.concatenate([
        e[:, u_idx] * w[u_idx],
        e[:, l_idx] * w[l_idx],
        pose / 6.0,
    ], axis=1).astype(np.float32)

    cand_idx = np.where(cand_mask)[0]
    if len(cand_idx) == 0:
        raise RuntimeError("no bank candidates after sharpness filter")

    # neighbour outlier rejection (on candidates only)
    keep = _neighbour_reject(feats[cand_idx], k=3, factor=3.0)
    cand_idx = cand_idx[keep]
    log.info("bank candidates after neighbour filter: %d", len(cand_idx))

    # 2. Quotas (§5.2 step 4)
    quotas: list[tuple[np.ndarray, int, str]] = []
    N_bank = int(cfg.bank.N)
    q_jaw = int(cfg.bank.quota_jaw_open)
    q_yaw = int(cfg.bank.quota_yaw_each_side)
    q_blink = int(cfg.bank.quota_blink)
    q_wide = int(cfg.bank.quota_wide)

    jaw_open_i = C.NAME_TO_IDX.get("jawOpen", -1)
    blink_l = C.NAME_TO_IDX.get("eyeBlinkLeft", -1)
    blink_r = C.NAME_TO_IDX.get("eyeBlinkRight", -1)
    wide_l = C.NAME_TO_IDX.get("eyeWideLeft", -1)
    wide_r = C.NAME_TO_IDX.get("eyeWideRight", -1)

    # Stage frame: highest-sharpness training frame whose e is close to neutral
    e0 = np.median(e[train_mask], axis=0)
    neutral_score = np.linalg.norm(e[cand_idx] - e0, axis=1)
    sharp_rank = -blur[cand_idx]
    stage_order = np.lexsort((sharp_rank, neutral_score))
    stage_frame = int(cand_idx[stage_order[0]])
    log.info("stage frame: row %d", stage_frame)

    selected: list[int] = [stage_frame]

    def _pick_quota(mask: np.ndarray, k: int, label: str) -> None:
        if k <= 0:
            return
        pool = cand_idx[mask[cand_idx]]
        pool = np.setdiff1d(pool, np.array(selected, dtype=np.int64), assume_unique=False)
        if len(pool) == 0:
            log.warning("quota '%s': no candidates", label)
            return
        seeds = feats[[stage_frame]]
        local = _fps_from_seed(feats[pool], k, seeds)
        picked = pool[local]
        selected.extend(int(x) for x in picked)
        log.info("quota %-14s picked %d", label, len(picked))

    if jaw_open_i >= 0:
        m = e[:, jaw_open_i] > 0.4
        _pick_quota(m, q_jaw, "jaw_open")
    if q_yaw > 0:
        for sign, tag in ((+1, "yaw_pos"), (-1, "yaw_neg")):
            m = (sign * pose[:, 0] > 15.0)
            _pick_quota(m, q_yaw, tag)
    if blink_l >= 0 and blink_r >= 0:
        m = ((e[:, blink_l] > 0.4) | (e[:, blink_r] > 0.4))
        _pick_quota(m, q_blink, "blink")
    if wide_l >= 0 and wide_r >= 0:
        m = ((e[:, wide_l] > 0.4) | (e[:, wide_r] > 0.4))
        _pick_quota(m, q_wide, "wide")

    # 3. FPS fill to N
    selected = sorted(set(selected))
    remaining = np.setdiff1d(cand_idx, np.array(selected, dtype=np.int64), assume_unique=False)
    n_fill = max(0, N_bank - len(selected))
    if n_fill > 0 and len(remaining) > 0:
        seeds = feats[np.array(selected, dtype=np.int64)]
        local = _fps_from_seed(feats[remaining], n_fill, seeds)
        selected.extend(int(x) for x in remaining[local])

    selected = sorted(set(selected))[:N_bank]
    selected_arr = np.array(selected, dtype=np.int64)
    log.info("bank selected %d / %d", len(selected_arr), N_bank)

    # 4. Save
    bank = {
        "rows": selected_arr,                    # indices into frames.npz / headcrops.npy
        "native_rows": nat_row[selected_arr],    # indices into nativecrops.npy (-1 if absent)
        "e": e[selected_arr],
        "pose_deg": pose[selected_arr],
        "split": split[selected_arr],
        "sharpness": blur[selected_arr],
        "session": d["session"][selected_arr],
        "native_side": int(d["native_side"]),
        "weights": w,                            # for §5.2 slot retrieval
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{k: v for k, v in bank.items() if isinstance(v, np.ndarray)})
    save_json({
        "n_selected": int(len(selected_arr)),
        "stage_frame_row": int(stage_frame),
        "weights": w.tolist(),
    }, out_path.with_suffix(".json"))
    return bank
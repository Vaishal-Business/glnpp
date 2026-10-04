import numpy as np
from types import SimpleNamespace
from pathlib import Path

from galnp_s.capture.bank_select import _fps_from_seed, _neighbour_reject, _weights_from_train
from galnp_s.capture import channels as C


def test_fps_from_seed_picks_diverse():
    rng = np.random.default_rng(0)
    # two clusters far apart
    a = rng.standard_normal((40, 3)) * 0.1
    b = rng.standard_normal((40, 3)) * 0.1 + 5.0
    feats = np.vstack([a, b]).astype(np.float32)
    seeds = feats[[0]]  # one seed in cluster A
    idx = _fps_from_seed(feats, 4, seeds)
    # at least one pick should be from cluster B (which is far from seed)
    assert any(i >= 40 for i in idx.tolist())


def test_neighbour_reject_isolates_outliers():
    rng = np.random.default_rng(0)
    cluster = rng.standard_normal((100, 3)) * 0.01
    outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    feats = np.vstack([cluster, outlier]).astype(np.float32)
    mask = _neighbour_reject(feats, k=3, factor=3.0)
    assert not mask[-1], "outlier should be rejected"
    assert mask[:-1].mean() > 0.9


def test_weights_boost_critical():
    e = np.zeros((100, 52), dtype=np.float32)
    e[:, C.NAME_TO_IDX["jawOpen"]] = np.linspace(0, 1, 100)
    e[:, C.NAME_TO_IDX["eyeBlinkLeft"]] = np.linspace(0, 1, 100)
    supported = np.ones(52, dtype=bool)
    w = _weights_from_train(e, supported)
    assert w[C.NAME_TO_IDX["jawOpen"]] > w[0]
    assert w[C.NAME_TO_IDX["eyeBlinkLeft"]] > w[0]
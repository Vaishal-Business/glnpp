#!/usr/bin/env python
"""S0 step 1: track a recording session and cache the tracker output.

Usage:
    python scripts/s0_track.py --subject configs/subject.yaml [--session 0]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from galnp_s.config import load_config
from galnp_s.capture.tracker import track_to_npz
from galnp_s.utils.logging import get_logger
from galnp_s.utils.seed import set_seed

log = get_logger("s0_track")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.base, args.subject)
    set_seed(cfg.seed)

    workdir = Path(cfg.workdir)
    for sess in cfg.sessions:
        if args.session is not None and int(sess.id) != args.session:
            continue
        out = workdir / f"session{sess.id}" / "tracker.npz"
        if out.exists():
            log.info("session %s already tracked -> %s (skip)", sess.id, out)
            continue
        log.info("=== session %s : %s ===", sess.id, sess.video)
        track_to_npz(sess.video, out, cfg.tracker_model)


if __name__ == "__main__":
    main()
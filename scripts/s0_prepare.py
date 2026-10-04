#!/usr/bin/env python
"""S0 end-to-end: prepare -> coverage report -> bank select.

Usage (after `s0_track.py` has populated `<workdir>/session<N>/tracker.npz`):

    python scripts/s0_prepare.py --subject configs/subject.yaml
    python scripts/s0_prepare.py --subject configs/subject.yaml --session 0
    python scripts/s0_prepare.py --subject configs/subject.yaml --skip-bank
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from galnp_s.capture.prepare import process_session
from galnp_s.capture.coverage_report import report_and_gate
from galnp_s.capture.bank_select import select_bank
from galnp_s.config import load_config
from galnp_s.utils.logging import get_logger
from galnp_s.utils.seed import set_seed

log = get_logger("s0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", type=int, default=None)
    ap.add_argument("--skip-bank", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.base, args.subject)
    set_seed(int(cfg.seed))
    workdir = Path(cfg.workdir)

    sessions = [s for s in cfg.sessions
                if args.session is None or int(s.id) == args.session]
    if not sessions:
        log.error("no matching sessions in subject config")
        return 2

    all_reports = []
    for sess in sessions:
        sid = int(sess.id)
        sdir = workdir / f"session{sid}"
        sdir.mkdir(parents=True, exist_ok=True)

        tracker_npz = sdir / "tracker.npz"
        if not tracker_npz.exists():
            log.error("missing %s (run scripts/s0_track.py first)", tracker_npz)
            return 3

        # --- prepare ---
        summary = process_session(sess.video, tracker_npz, sdir, sid, sess, cfg)

        # --- coverage + G0 ---
        # tracker valid ratio from the tracker npz
        import numpy as np
        tr = np.load(tracker_npz)
        tv_ratio = float(tr["valid"].mean())

        report, ok, fails = report_and_gate(
            sdir / "frames.npz", subject_id=str(cfg.get("subject_id", "subject")),
            cfg=cfg, tracker_valid_ratio=tv_ratio, out_dir=sdir,
        )
        all_reports.append((sid, ok, fails))

        if not ok:
            log.error("=== S0 for session %d FAILED G0. Do NOT train. ===", sid)
            for m in fails:
                log.error("  - %s", m)
            return 4

        # --- bank ---
        if not args.skip_bank:
            select_bank(
                frames_npz=sdir / "frames.npz",
                coverage_json=sdir / "coverage.json",
                cfg=cfg,
                out_path=sdir / "bank.npz",
            )

    log.info("=== all sessions passed G0 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
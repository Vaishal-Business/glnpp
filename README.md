# GALNP-S

Single-Identity Geometry-Anchored Laplacian Neural Painter.
Specification: see `GALNP-S Single-Identity Geometry-Anchored.txt`.

## Status

- [x] Step 0 — scaffold, config, tracker wrapper
- [x] Step 1 — capture/prepare (S0), coverage report, bank select
- [ ] Step 2 — geometry closed-form init (S1)
- [ ] Step 3 — rasterizer + tests
- [ ] Step 4 — bank, neural texture (S2, needs T4)
- [ ] Step 5 — painters, losses (S3–S6, needs T4)
- [ ] Step 6 — adapter, support sets, export (S7)

## S0 (first run on your recording)

1. `cp configs/subject.example.yaml configs/subject.yaml` and edit.
2. `python scripts/s0_track.py --subject configs/subject.yaml`
3. `python scripts/s0_prepare.py --subject configs/subject.yaml`

Step 3 runs prepare → coverage report → G0 gate → bank selection.
If G0 fails, **do not train**; top up the recording instead.

## Local prerequisites

- Python ≥ 3.10
- `ffmpeg` on PATH (OpenCV decode fallback)
- ~4 GB free disk per session of 1080p/30fps video

## Setup (local, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
pytest -q
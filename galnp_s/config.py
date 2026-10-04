"""Config loading: base.yaml -> subject.yaml -> stage override, with dot access.

Usage:
    from galnp_s.config import load_config
    cfg = load_config("configs/base.yaml", "configs/subject.yaml", "s0")
    cfg.bank.N            # 96
    cfg.capture.face_oval_min_px   # 450
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig


def _read_yaml(path: str | Path) -> DictConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return OmegaConf.load(p)


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Merge configs left-to-right; later files win. `overrides` are dotlist strings."""
    if not paths:
        raise ValueError("load_config needs at least one path")
    cfg = _read_yaml(paths[0])
    for p in paths[1:]:
        cfg = OmegaConf.merge(cfg, _read_yaml(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def dump_config(cfg: DictConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path), resolve=True)
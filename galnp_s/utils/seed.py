from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int = 1234) -> None:
    """Deterministic seeding for every RNG we touch. See §18 (`seed: 1234`)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism is *required* (F23). Do not flip these.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
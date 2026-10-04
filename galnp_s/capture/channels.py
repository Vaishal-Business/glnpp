"""Channel vocabulary for the tracker's 52-d blendshape vector.

`CHANNEL_NAMES` follows the MediaPipe Face Landmarker order (ARKit-style names
plus a `_neutral` placeholder at index 0 which is ignored downstream).

Groups U and L are defined in §6.2. `cheekSquintL/R` intentionally live in both.
"""
from __future__ import annotations

CHANNEL_NAMES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)
assert len(CHANNEL_NAMES) == 52, len(CHANNEL_NAMES)

NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(CHANNEL_NAMES)}

# --- groups (§6.2) -----------------------------------------------------------
UPPER_GROUP: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "cheekSquintLeft", "cheekSquintRight",
)                                                                  # 21
LOWER_GROUP: tuple[str, ...] = (
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
)                                                                  # 32
# `tongueOut` is optional: it is present in some builds. Detected at S0 and
# appended to LOWER_GROUP if found; otherwise silently ignored.

CRITICAL_CHANNELS: tuple[str, ...] = (
    "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeWideLeft", "eyeWideRight",
    "browInnerUp",
    "browDownLeft", "browDownRight",
    "cheekSquintLeft", "cheekSquintRight",
)


def idx(name: str) -> int:
    return NAME_TO_IDX[name]


def indices(names) -> list[int]:
    return [NAME_TO_IDX[n] for n in names]


def upper_indices(extra: tuple[str, ...] = ()) -> list[int]:
    return indices(UPPER_GROUP + tuple(extra))


def lower_indices(extra: tuple[str, ...] = ()) -> list[int]:
    return indices(LOWER_GROUP + tuple(extra))
#nothin
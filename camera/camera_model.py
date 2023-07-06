from dataclasses import dataclass


@dataclass
class CameraModel:
    """Camera model parameters."""

    # H, W, cx, cy, fx, fy, zn, zf
    # resolutions
    H: int
    W: int

    # camera intrinsics
    cx: float
    cy: float
    fx: float
    fy: float

    # z-culling
    zn: float
    zf: float

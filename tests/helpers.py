import numpy as np
from vision.types import HandData

_MCP = {"index": (-0.3, -0.95), "middle": (0.0, -1.0), "ring": (0.3, -0.95), "pinky": (0.55, -0.85)}
_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(thumb="fold", index=False, middle=False, ring=False, pinky=False, scale=0.2, rot=0.0,
              offset=(0.5, 0.6), label="Right"):
    """Synthetic 21-landmark hand in normalized image coords (y down). thumb: out|up|fold."""
    p = np.zeros((21, 3))
    p[1], p[2] = (-0.25, -0.3, 0), (-0.5, -0.5, 0)
    if thumb == "out":
        p[3], p[4] = (-0.8, -0.75, 0), (-1.05, -1.0, 0)
    elif thumb == "up":
        p[2] = (-0.4, -0.5, 0); p[3] = (-0.45, -0.85, 0); p[4] = (-0.55, -1.7, 0)
    else:
        p[2] = (-0.45, -0.5, 0); p[3] = (-0.3, -0.65, 0); p[4] = (-0.05, -0.6, 0)
    for name, ext in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        b = _BASE[name]
        x, y = _MCP[name]
        p[b] = (x, y, 0)
        if ext:
            p[b + 1], p[b + 2], p[b + 3] = (x, y - 0.4, 0), (x, y - 0.68, 0), (x, y - 0.9, 0)
        else:
            p[b + 1], p[b + 2], p[b + 3] = (x, y - 0.3, 0), (x, y - 0.05, 0), (x, y + 0.2, 0)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    p[:, :2] = p[:, :2] @ R.T
    lm = p * scale
    lm[:, 0] += offset[0]
    lm[:, 1] += offset[1]
    lm = lm.astype(np.float32)
    return HandData(label, lm, lm[:, :2] * [640, 360], 0.95)


def open_hand(**k):
    return make_hand(thumb="out", index=True, middle=True, ring=True, pinky=True, **k)


def fist(**k):
    return make_hand(**k)

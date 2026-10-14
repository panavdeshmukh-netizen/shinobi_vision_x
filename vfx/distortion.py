import cv2
import numpy as np
from .base import Effect
from .compositing import ease_out

_GRID = {}


def _grid(h, w):
    if (h, w) not in _GRID:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID[(h, w)] = (xs, ys)
    return _GRID[(h, w)]


class Distortion(Effect):
    name, label, title, duration = "distortion", "Space Distortion", "SPACE WARP", 2.0
    layer = "frame"

    def pick_anchor(self, ctx):
        return ctx.center_anchor() if len(ctx.hands) > 1 else ctx.hand_anchor()

    def apply(self, frame, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        h, w = frame.shape[:2]
        R = int(min(w, h) * 0.38)
        cx, cy = int(self.pos[0]), int(self.pos[1])
        x0, y0, x1, y1 = max(cx - R, 0), max(cy - R, 0), min(cx + R, w), min(cy + R, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        xs, ys = _grid(h, w)
        dx, dy = xs[y0:y1, x0:x1] - cx, ys[y0:y1, x0:x1] - cy
        r = np.sqrt(dx * dx + dy * dy)
        ang = 4.5 * env * np.clip(1 - r / R, 0, 1) ** 2
        c, sn = np.cos(ang), np.sin(ang)
        mx, my = cx + dx * c - dy * sn, cy + dx * sn + dy * c
        frame[y0:y1, x0:x1] = cv2.remap(frame, mx.astype(np.float32), my.astype(np.float32),
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

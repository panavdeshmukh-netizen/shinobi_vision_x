"""Blending helpers: additive, alpha, bloom, sprites, colour grade."""
from functools import lru_cache
import cv2
import numpy as np


def alpha_over(dst, src_bgra):
    """Alpha-composite a BGRA layer over a BGR frame in place."""
    a = src_bgra[..., 3:4].astype(np.float32) / 255.0
    out = dst.astype(np.float32) * (1 - a) + src_bgra[..., :3].astype(np.float32) * a
    dst[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return dst


def additive(dst, src, k=1.0):
    cv2.addWeighted(dst, 1.0, src, float(k), 0, dst)
    return dst


def bloom(layer, strength=0.6, scale=4):
    """Return layer + soft glow (blur at reduced resolution => cheap)."""
    if strength <= 0.01:
        return layer
    h, w = layer.shape[:2]
    small = cv2.resize(layer, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(layer, 1.0, glow, float(strength) * 2.0, 0)


@lru_cache(maxsize=96)
def _sprite(radius, color):
    r = max(2, int(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / r
    f = np.clip(1 - d, 0, 1) ** 2
    return np.clip(f[..., None] * np.array(color, np.float32)[None, None, :], 0, 255).astype(np.uint8)


def radial_sprite(radius, color):
    """Cached soft radial gradient (BGR, additive use). Radius quantised to 4 px."""
    return _sprite(max(4, int(round(radius / 4.0)) * 4), tuple(int(c) for c in color))


def blit_add(canvas, sprite, center, k=1.0):
    h, w = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = int(center[0]) - sw // 2, int(center[1]) - sh // 2
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + sw, w), min(y0 + sh, h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    src = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    dst = canvas[cy0:cy1, cx0:cx1]
    cv2.addWeighted(dst, 1.0, src, float(k), 0, dst)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def sc(color, k):
    k = max(0.0, min(float(k), 1.0))
    return tuple(int(c * k) for c in color)


_LUT = None


def color_grade(frame, amount=0.3):
    """Mild cinematic grade: contrast curve + cool shadows."""
    global _LUT
    if _LUT is None:
        x = np.arange(256, dtype=np.float32) / 255
        s = x * x * (3 - 2 * x)
        _LUT = (np.clip(s, 0, 1) * 255).astype(np.uint8)
    graded = cv2.LUT(frame, _LUT)
    graded[..., 0] = cv2.add(graded[..., 0], 6)
    return cv2.addWeighted(frame, 1 - amount, graded, amount, 0)

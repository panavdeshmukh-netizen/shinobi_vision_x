"""Approximate shadow-clone effect built from buffered frames ("frame echo").

No person segmentation is used: clones are time-shifted, horizontally offset, faded copies of
the whole frame. It reads as clones on plain backgrounds but is NOT perfect background removal.
"""
from collections import deque
import cv2
import numpy as np
from .base import Effect


class Clones(Effect):
    name, label, title, duration = "clones", "Shadow Clones (frame echo)", "SHADOW ECHO", 4.0
    layer = "frame"

    def __init__(self, maxlen=24):
        super().__init__()
        self.buf = deque(maxlen=maxlen)   # half-resolution frames => low memory
        self.n = 6
        self.gap = 3

    def push(self, frame):
        h, w = frame.shape[:2]
        self.buf.append(cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA))

    def apply(self, frame, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02 or len(self.buf) < 4:
            return
        h, w = frame.shape[:2]
        n = max(2, int(self.n * (0.5 if s.performance else 1.0)))
        for k in range(n, 0, -1):
            i = max(0, len(self.buf) - 1 - k * self.gap)
            old = cv2.resize(self.buf[i], (w, h), interpolation=cv2.INTER_LINEAR)
            side = -1 if k % 2 else 1
            dx = side * ((k + 1) // 2) * 0.15 * w
            M = np.float32([[0.95, 0, dx + 0.025 * w], [0, 0.95, 0.04 * h]])
            sh = cv2.warpAffine(old, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            a = 0.42 * env * (1 - (k - 1) / (n + 1))
            cv2.addWeighted(frame, 1 - a, sh, a, 0, frame)

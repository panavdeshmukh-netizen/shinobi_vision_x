import cv2
import numpy as np
from .base import Effect
from .compositing import ease_out, sc
from .particles import ParticleSystem


def bolt(p0, p1, rng, depth=4, disp=0.22):
    pts = [np.asarray(p0, float), np.asarray(p1, float)]
    for _ in range(depth):
        new = []
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            new += [a, (a + b) / 2 + nrm * rng.normal(0, disp * np.linalg.norm(d))]
        new.append(pts[-1])
        pts = new
    return np.array(pts, np.int32)


class Lightning(Effect):
    name, label, title, duration = "lightning", "Lightning", "LIGHTNING SURGE", 3.0
    palette = ((255, 150, 40), (255, 255, 255))

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(260)

    def on_start(self, ctx):
        self.ps.clear()

    def on_update(self, ctx):
        s = getattr(self, "_s", None)
        ps_scale = s.pscale if s else 1.0
        self.ps.emit(int(5 * ps_scale * self.env) + 1, self.pos, speed=(40, 220), life=(0.2, 0.6),
                     size=(1, 3), color=(255, 230, 160), spread=self.scale * 0.2)
        self.ps.update(ctx.dt, drag=2.0)

    def draw(self, canvas, ctx, s):
        self._s = s
        env = self.env * s.intensity
        if env <= 0.02:
            return
        R = self.scale * (0.9 + 1.5 * ease_out(self.t / 0.5))
        n = int(6 * s.pscale * env) + 2
        for _ in range(n):
            a = self.rng.uniform(0, 6.283)
            end = self.pos + np.array([np.cos(a), np.sin(a)]) * R * self.rng.uniform(0.4, 1.1)
            pts = bolt(self.pos, end, self.rng)
            k = self.rng.uniform(0.55, 1.0) * min(env, 1.0)
            cv2.polylines(canvas, [pts], False, sc(self.palette[0], k), 4, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], False, sc(self.palette[1], k), 1, cv2.LINE_AA)
        self.ps.draw(canvas, boost=min(env, 1.2))


class Chidori(Lightning):
    name, label, title, duration = "chidori", "Chidori", "CHIDORI", 2.1
    palette = ((255, 210, 70), (255, 255, 245))

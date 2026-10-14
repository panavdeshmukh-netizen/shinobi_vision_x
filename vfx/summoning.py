import cv2
import numpy as np
from .base import Effect
from .compositing import ease_out, sc
from .particles import ParticleSystem


class Summoning(Effect):
    name, label, title, duration = "summoning", "Summoning Circle", "SUMMONING", 4.0
    SQUASH = 0.32

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(200)

    def pick_anchor(self, ctx):
        c = ctx.center_anchor() if len(ctx.hands) > 1 else ctx.hand_anchor()
        return None if c is None else np.array([c[0], ctx.h * 0.8])

    def on_update(self, ctx):
        a = self.rng.uniform(0, 6.283)
        R = ctx.w * 0.22 * ease_out(self.t / 0.6)
        p = self.pos + np.array([np.cos(a) * R, np.sin(a) * R * self.SQUASH])
        self.ps.emit(3, p, speed=(20, 60), angle=(-1.8, -1.3), life=(0.5, 1.2), size=(1, 3), color=(255, 170, 90))
        self.ps.update(ctx.dt)

    def _ell(self, canvas, R, a0, a1, rot, col, th=2):
        cv2.ellipse(canvas, (int(self.pos[0]), int(self.pos[1])), (max(2, int(R)), max(2, int(R * self.SQUASH))),
                    0, a0, a1, col, th, cv2.LINE_AA)

    def draw(self, canvas, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        R = ctx.w * 0.22 * ease_out(self.t / 0.6)
        rot = self.t * 40
        col = sc((255, 170, 70), env)
        for f, th in ((1.0, 3), (0.9, 1), (0.6, 2), (0.3, 1)):
            self._ell(canvas, R * f, 0, 360, 0, col, th)
        for i in range(12):  # rotating dashes (original geometric glyph ring)
            a = rot + i * 30
            self._ell(canvas, R * 0.78, a, a + 14, 0, sc((255, 220, 140), env), 3)
        for tri in (0, 60):  # star (two triangles) projected onto the ground plane
            pts = []
            for j in range(3):
                th = np.radians(tri + j * 120 + rot * 1.5)
                pts.append([self.pos[0] + np.cos(th) * R * 0.6, self.pos[1] + np.sin(th) * R * 0.6 * self.SQUASH])
            cv2.polylines(canvas, [np.array(pts, np.int32)], True, col, 2, cv2.LINE_AA)
        self.ps.draw(canvas, boost=env)

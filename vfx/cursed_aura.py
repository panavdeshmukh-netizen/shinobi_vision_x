"""Red aura. Approximate: anchored to the frame centre/upper body (no body segmentation);
eye glow via face landmarks is NOT implemented."""
import cv2
import numpy as np
from .base import Effect
from .compositing import sc
from .particles import ParticleSystem


class CursedAura(Effect):
    name, label, title, duration = "cursed_aura", "Cursed Aura (approx.)", "CURSED AURA", 4.0

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(240)

    def pick_anchor(self, ctx):
        c = ctx.center_anchor()
        return np.array([c[0] if c is not None else ctx.w / 2, ctx.h * 0.62])

    def on_update(self, ctx):
        s = getattr(self, "_s", None)
        pk = s.pscale if s else 1.0
        for _ in range(int(4 * pk) + 1):
            x = self.pos[0] + self.rng.normal(0, ctx.w * 0.14)
            self.ps.emit(1, (x, self.pos[1] + ctx.h * 0.2), speed=(10, 40), angle=(-1.9, -1.2),
                         life=(0.6, 1.5), size=(2, 4), color=(30, 60, 255))
        self.ps.update(ctx.dt, gravity=(0, -30))

    def draw(self, canvas, ctx, s):
        self._s = s
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        flick = 0.75 + 0.25 * np.sin(self.t * 17) + self.rng.uniform(-0.05, 0.05)
        c = (int(self.pos[0]), int(self.pos[1]))
        for f, col in ((1.0, (20, 30, 150)), (0.85, (30, 70, 230)), (0.7, (40, 120, 255))):
            cv2.ellipse(canvas, c, (int(ctx.w * 0.17 * f), int(ctx.h * 0.42 * f)), 0, 0, 360,
                        sc(col, env * flick * 0.55), 6, cv2.LINE_AA)
        self.ps.draw(canvas, boost=env)

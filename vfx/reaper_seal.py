import cv2
import numpy as np
from .base import Effect
from .compositing import ease_out, sc
from .particles import ParticleSystem


class ReaperSeal(Effect):
    name, label, title, duration = "reaper_seal", "Reaper Seal", "REAPER SEAL", 4.5

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(200)

    def pick_anchor(self, ctx):
        return np.array([ctx.w / 2, ctx.h * 0.45])

    def on_update(self, ctx):
        a = self.rng.uniform(0, 6.283)
        R = ctx.h * 0.4
        self.ps.emit(3, self.pos + np.array([np.cos(a), np.sin(a)]) * R, speed=(5, 25),
                     life=(0.8, 1.8), size=(3, 6), color=(110, 80, 110))
        self.ps.update(ctx.dt, gravity=(0, -15))

    def draw(self, canvas, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        R = ctx.h * 0.4 * ease_out(self.t / 0.7)
        c = (int(self.pos[0]), int(self.pos[1]))
        for f, th in ((1.0, 3), (0.93, 1), (0.62, 2), (0.3, 1)):
            cv2.circle(canvas, c, max(2, int(R * f)), sc((190, 90, 200), env), th, cv2.LINE_AA)
        for k, direction in ((0, 1), (60, 1)):
            pts = [[c[0] + np.cos(np.radians(k + j * 120 + direction * self.t * 30)) * R * 0.6,
                    c[1] + np.sin(np.radians(k + j * 120 + direction * self.t * 30)) * R * 0.6] for j in range(3)]
            cv2.polylines(canvas, [np.array(pts, np.int32)], True, sc((230, 130, 255), env), 2, cv2.LINE_AA)
        for i in range(36):  # rotating tick marks
            a = np.radians(i * 10 - self.t * 25)
            p0 = (c[0] + np.cos(a) * R * 0.95, c[1] + np.sin(a) * R * 0.95)
            p1 = (c[0] + np.cos(a) * R * (1.08 if i % 3 == 0 else 1.02), c[1] + np.sin(a) * R * (1.08 if i % 3 == 0 else 1.02))
            cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), sc((200, 100, 220), env), 2, cv2.LINE_AA)
        self.ps.draw(canvas, boost=env * 0.7)

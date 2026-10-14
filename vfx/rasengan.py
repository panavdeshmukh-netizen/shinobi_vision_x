import cv2
import numpy as np
from .base import Effect
from .compositing import blit_add, ease_out, radial_sprite, sc
from .particles import ParticleSystem


class Rasengan(Effect):
    name, label, title, duration, forward = "rasengan", "Energy Orb", "ENERGY ORB", 3.2, 0.0

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(200)
        self.launched = False
        self.launch_direction = np.array([0.0, -1.0])

    def on_start(self, ctx):
        self.launched = False
        self.launch_direction = np.array([0.0, -1.0])
        self.ps.clear()

    def launch(self, ctx):
        if not self.active or self.launched:
            return False
        hand = next((h for h in ctx.hands if h.label == self.hand_label), None)
        if hand is None:
            return False
        self.launch_direction = hand.direction.copy()
        self.hand_label = None
        self.launched = True
        return True

    def pick_anchor(self, ctx):
        return None if self.launched else super().pick_anchor(ctx)

    def on_update(self, ctx):
        if self.launched:
            self.pos += self.launch_direction * ctx.w * 0.58 * ctx.dt
        a = self.rng.uniform(0, 6.283)
        r = self.scale * 0.8
        self.ps.emit(2, self.pos + np.array([np.cos(a), np.sin(a)]) * r, speed=(5, 30), life=(0.3, 0.7),
                     size=(1, 2), color=(255, 220, 150))
        self.ps.update(ctx.dt, attract=self.pos, k=2.0)

    def draw(self, canvas, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        R = self.scale * 0.75 * ease_out(self.t / 0.4)
        burst = max(0.0, self.t - (self.duration - 0.5)) / 0.5
        c = (int(self.pos[0]), int(self.pos[1]))
        blit_add(canvas, radial_sprite(R * 2.0, (255, 110, 25)), c, env)
        blit_add(canvas, radial_sprite(R * 1.15, (255, 220, 75)), c, env)
        blit_add(canvas, radial_sprite(R * 0.48, (255, 255, 235)), c, env)
        for i in range(8):
            ang = self.t * (420 if i % 2 else -330) + i * 45
            cv2.ellipse(canvas, c, (max(2, int(R * (0.30 + 0.085 * i))), max(2, int(R * (0.18 + 0.07 * i)))),
                        ang, 12, 264, sc((255, 190 + 8 * (i % 4), 70 + 18 * (i % 3)), env), 2, cv2.LINE_AA)
        if burst > 0:
            cv2.circle(canvas, c, int(R * (1 + 2.5 * burst)), sc((255, 220, 160), env * (1 - burst)), 3, cv2.LINE_AA)
        self.ps.draw(canvas, boost=env)

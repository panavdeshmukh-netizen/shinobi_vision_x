import cv2
import numpy as np
from .base import Effect
from .compositing import blit_add, ease_out, radial_sprite, sc
from .particles import ParticleSystem


class Awakening(Effect):
    """Both open palms -> synchronised energy burst between the hands."""
    name, label, title, duration = "awakening", "Twin Palm Awakening", "TWIN PALM AWAKENING", 3.0

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(260)

    def pick_anchor(self, ctx):
        return ctx.center_anchor()

    def on_start(self, ctx):
        self.ps.clear()
        self.ps.emit(120, self.pos, speed=(80, 380), life=(0.5, 1.2), size=(2, 4), color=(60, 80, 255))

    def on_update(self, ctx):
        self.ps.update(ctx.dt, drag=1.5)

    def draw(self, canvas, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        if env <= 0.02:
            return
        c = (int(self.pos[0]), int(self.pos[1]))
        blit_add(canvas, radial_sprite(self.scale * 2.2, (40, 40, 255)), c, env)
        for i in range(3):
            tt = (self.t * 1.4 + i / 3) % 1.0
            cv2.circle(canvas, c, int(self.scale * (0.5 + 4 * tt)), sc((90, 90, 255), env * (1 - tt)), 3, cv2.LINE_AA)
        for h in ctx.hands:
            blit_add(canvas, radial_sprite(h.scale * 0.9, (60, 100, 255)), h.palm_center, env)
        self.ps.draw(canvas, boost=env)

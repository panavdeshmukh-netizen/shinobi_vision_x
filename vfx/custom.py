import cv2
import numpy as np
from .base import Effect
from .compositing import blit_add, radial_sprite, sc
from .particles import ParticleSystem


class CustomEffect(Effect):
    """Configurable effect: colour (BGR), particle count, duration, glow strength, scale."""
    name, label, title = "custom", "Custom VFX", "CUSTOM"

    def __init__(self, color=(255, 120, 255), particles=120, duration=2.5, glow=1.0, scale=1.0):
        super().__init__()
        self.color, self.particles, self.duration, self.glow, self.k = color, particles, duration, glow, scale
        self.ps = ParticleSystem(300)

    def configure(self, d):
        self.color = tuple(d.get("color", self.color))
        self.particles = int(d.get("particles", self.particles))
        self.duration = float(d.get("duration", self.duration))
        self.glow = float(d.get("glow", self.glow))
        self.k = float(d.get("scale", self.k))

    def on_start(self, ctx):
        self.ps.clear()
        self.ps.emit(min(self.particles, 300), self.pos, speed=(30, 220 * self.k), life=(0.5, 1.3), size=(1, 4), color=self.color)

    def on_update(self, ctx):
        self.ps.update(ctx.dt, drag=1.0)

    def draw(self, canvas, ctx, s):
        env = min(self.env, 1.0) * s.intensity
        blit_add(canvas, radial_sprite(self.scale * 1.6 * self.k, self.color), self.pos, env * self.glow)
        cv2.circle(canvas, (int(self.pos[0]), int(self.pos[1])), int(self.scale * (0.6 + self.t) * self.k),
                   sc(self.color, env), 2, cv2.LINE_AA)
        self.ps.draw(canvas, boost=env)

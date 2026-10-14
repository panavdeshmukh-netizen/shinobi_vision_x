import numpy as np
from .base import Effect
from .compositing import blit_add, radial_sprite
from .particles import ParticleSystem


class Fireball(Effect):
    name, label, title, duration, forward = "fireball", "Fireball", "FIRE BURST", 2.8, 0.4
    charge = 0.9

    def __init__(self):
        super().__init__()
        self.ps = ParticleSystem(320)
        self.fpos = None
        self.vel = None

    def on_start(self, ctx):
        self.fpos, self.vel = None, None
        self.manual_charge = False
        self.launched = False
        self.ps.clear()

    def hold_charge(self):
        self.manual_charge = True

    def launch(self, ctx):
        if not self.active or self.launched:
            return False
        hand = next((h for h in ctx.hands if h.label == self.hand_label), None)
        direction = hand.direction if hand is not None else np.array([0.0, -1.0])
        self.fpos = self.pos.copy()
        self.vel = direction * ctx.w * 0.5
        self.launched = True
        return True

    def on_update(self, ctx):
        s = getattr(self, "_s", None)
        pk = s.pscale if s else 1.0
        if not self.launched and (self.manual_charge or self.t < self.charge):
            a = self.rng.uniform(0, 6.283, 4)
            for ang in a[: max(1, int(3 * pk))]:
                p = self.pos + np.array([np.cos(ang), np.sin(ang)]) * self.scale * 1.2
                self.ps.emit(1, p, speed=(0, 10), life=(0.3, 0.6), size=(2, 4), color=(30, 120, 255))
            self.ps.update(ctx.dt, attract=self.pos, k=4.0)
        elif not self.launched:
            self.launch(ctx)
        else:
            self.fpos = self.fpos + self.vel * ctx.dt
            self.ps.emit(int(8 * pk) + 1, self.fpos, speed=(10, 80), life=(0.3, 0.8), size=(2, 5),
                         color=(40, 140, 255), spread=self.scale * 0.2)
            self.ps.update(ctx.dt, gravity=(0, -60), drag=1.0)

    def draw(self, canvas, ctx, s):
        self._s = s
        k = min(self.env, 1.0) * s.intensity
        p = self.pos if self.fpos is None else self.fpos
        r = self.scale * (0.3 + 0.7 * min(1.0, self.t / self.charge)) if self.fpos is None else self.scale * 1.0
        blit_add(canvas, radial_sprite(r * 2, (20, 90, 255)), p, k)
        blit_add(canvas, radial_sprite(r, (120, 220, 255)), p, k)
        self.ps.draw(canvas, boost=k)

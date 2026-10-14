from dataclasses import dataclass, field
import numpy as np


@dataclass
class VFXSettings:
    intensity: float = 1.0
    particle_density: float = 1.0
    glow: float = 0.6
    trail: float = 0.4
    screen_shake: bool = False
    color_grade: bool = False
    performance: bool = False

    @property
    def pscale(self):
        return max(0.05, self.particle_density * (0.5 if self.performance else 1.0))


@dataclass
class FrameContext:
    w: int
    h: int
    hands: list = field(default_factory=list)
    dt: float = 0.033

    def hand_anchor(self, forward=0.0, label=None):
        hands = [hand for hand in self.hands if label is None or hand.label == label]
        if not hands:
            return None
        hnd = hands[0]
        return hnd.palm_center + hnd.direction * hnd.scale * forward

    def center_anchor(self):
        if not self.hands:
            return None
        return np.mean([h.palm_center for h in self.hands], axis=0)

    def hand_scale(self, label=None):
        hands = [hand for hand in self.hands if label is None or hand.label == label]
        if not hands:
            return None
        return float(np.mean([h.scale for h in hands]))


class Effect:
    name = "effect"
    label = "Effect"
    title = ""
    duration = 2.5
    layer = "additive"   # "additive" draws to glow canvas, "frame" edits the video frame
    forward = 0.0

    def __init__(self):
        self.t = 0.0
        self.active = False
        self.pos = None
        self.scale = 60.0
        self.hand_label = None
        self.rng = np.random.default_rng()

    # -- lifecycle -------------------------------------------------------
    def start(self, ctx, hand_label=None):
        self.t, self.active, self.pos = 0.0, True, None
        self.hand_label = hand_label
        self._track(ctx)
        self.on_start(ctx)

    def pick_anchor(self, ctx):
        return ctx.hand_anchor(self.forward, self.hand_label)

    def _track(self, ctx):
        a = self.pick_anchor(ctx)
        if a is not None:
            self.pos = a.copy() if self.pos is None else self.pos * 0.4 + a * 0.6
        elif self.pos is None:
            self.pos = np.array([ctx.w / 2, ctx.h / 2], float)
        s = ctx.hand_scale(self.hand_label)
        self.scale = self.scale * 0.8 + (s if s else ctx.h * 0.14) * 0.2 if self.t > 0 else (s if s else ctx.h * 0.14)

    def step(self, ctx):
        self.t += ctx.dt
        self._track(ctx)
        self.on_update(ctx)
        if self.t >= self.duration:
            self.active = False

    @property
    def env(self):
        return float(max(0.0, min(1.0, self.t / 0.25, (self.duration - self.t) / 0.45)))

    # -- hooks -----------------------------------------------------------
    def on_start(self, ctx): ...
    def on_update(self, ctx): ...
    def draw(self, canvas, ctx, s): ...
    def apply(self, frame, ctx, s): ...

"""Owns all effects, composites them over the live frame (single pixel coordinate system)."""
import cv2
import numpy as np
from .base import FrameContext, VFXSettings
from .compositing import bloom, color_grade
from .awakening import Awakening
from .clones import Clones
from .cursed_aura import CursedAura
from .custom import CustomEffect
from .distortion import Distortion
from .fireball import Fireball
from .lightning import Chidori, Lightning
from .rasengan import Rasengan
from .reaper_seal import ReaperSeal
from .summoning import Summoning


class VFXRenderer:
    def __init__(self, settings=None):
        self.settings = settings or VFXSettings()
        self.effects = {e.name: e for e in (Lightning(), Chidori(), Rasengan(), Fireball(), Clones(), Summoning(),
                                            CursedAura(), ReaperSeal(), Awakening(), Distortion(), CustomEffect())}
        self._canvas = None
        self._trail = None
        self._idle = 999
        self.flash = 0.0
        self.title, self.title_t = "", 0.0

    @property
    def names(self):
        return list(self.effects)

    def active_names(self):
        return [n for n, e in self.effects.items() if e.active]

    def trigger(self, name, ctx, hand_label=None):
        e = self.effects.get(name)
        if e is None:
            return False
        e.start(ctx, hand_label)
        self.flash = 1.0
        self.title, self.title_t = e.title, 1.6
        return True

    def _buf(self, frame):
        if self._canvas is None or self._canvas.shape != frame.shape:
            self._canvas = np.zeros_like(frame)
            self._trail = np.zeros_like(frame)
        return self._canvas

    def render(self, frame, ctx):
        s = self.settings
        canvas = self._buf(frame)
        clones = self.effects["clones"]
        clones.push(frame)
        for e in self.effects.values():
            if e.active:
                e.step(ctx)
        active = [e for e in self.effects.values() if e.active]
        for e in active:
            if e.layer == "frame":
                e.apply(frame, ctx, s)
        add = [e for e in active if e.layer == "additive"]
        if add or self._idle < 40:
            canvas[:] = 0
            for e in add:
                e.draw(canvas, ctx, s)
            decay = min(0.88, s.trail * 0.95)
            cv2.addWeighted(self._trail, decay, canvas, 1.0, 0, self._trail)
            layer = self._trail if s.trail > 0.02 else canvas
            self._idle = 0 if add else self._idle + 1
            if layer.any():
                cv2.add(frame, bloom(layer, s.glow, 6 if s.performance else 4), frame)
        else:
            self._idle += 1
        if active and s.color_grade:
            frame[:] = color_grade(frame, 0.3)
        if self.flash > 0:  # short, capped impact flash (never hides the video)
            a = 0.16 * self.flash
            cv2.addWeighted(frame, 1 - a, np.full_like(frame, 255), a, 0, frame)
            self.flash = max(0.0, self.flash - ctx.dt / 0.18)
        if active and s.screen_shake:
            amp = 4 * max(e.env for e in active)
            M = np.float32([[1, 0, np.random.uniform(-amp, amp)], [0, 1, np.random.uniform(-amp, amp)]])
            frame[:] = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        if self.title_t > 0:
            self._draw_title(frame)
            self.title_t -= ctx.dt
        return frame

    def _draw_title(self, frame):
        h, w = frame.shape[:2]
        scale = w / 640 * 1.2
        (tw, th), _ = cv2.getTextSize(self.title, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        org = ((w - tw) // 2, int(h * 0.86))
        cv2.putText(frame, self.title, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, self.title, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 200, 90), 3, cv2.LINE_AA)

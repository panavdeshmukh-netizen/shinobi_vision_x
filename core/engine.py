"""Frame pipeline: mirror -> resize -> track -> classify -> sequences -> VFX -> overlays."""
import time
import cv2
import numpy as np

from vision.gesture_features import feature_vector
from vision.face_tracker import FaceProximityController
from vfx.base import FrameContext
from vision.simple_controls import SimpleGestureController
from . import overlay

MODES = ("practice", "cinematic", "sequence", "sandbox", "calibration", "performance")


class Pipeline:
    def __init__(self, settings, tracker, classifier, sequences, renderer, calibration=None, face_tracker=None):
        self.settings, self.tracker, self.classifier = settings, tracker, classifier
        self.sequences, self.renderer, self.calibration = sequences, renderer, calibration
        self.face_tracker = face_tracker
        self.face_enabled = bool(settings.get("face_proximity_enabled", True)) and face_tracker is not None
        self.face_proximity = FaceProximityController(threshold=settings.get("face_proximity_threshold", 0.68))
        self.face_status = {"available": self.face_tracker is not None, "tracking": False, "face_found": False,
                    "near": {"Left": False, "Right": False},
                    "distance": {"Left": None, "Right": None}}
        self.fireball_charge_label = None
        self.face_cooldown_until = 0.0
        self.mode = "practice"
        self.set_mode(settings.get("mode", "practice"))
        self.control_mode = settings.get("control_mode", "simple")
        self.simple_enabled = bool(settings.get("simple_enabled", True))
        self.power_enabled = {
            "Right": bool(settings.get("rasengan_enabled", True)),
            "Left": bool(settings.get("chidori_enabled", True)),
            "Fireball": bool(settings.get("fireball_enabled", True)),
        }
        self.fireball_proximity_only = bool(settings.get("fireball_proximity_only", False))
        self.rasengan_motion_launch = bool(settings.get("rasengan_motion_launch", False))
        self.chidori_motion_discharge = bool(settings.get("chidori_motion_discharge", False))
        self.simple_controls = SimpleGestureController(cooldown=settings.get("jutsu_cooldown", 0.65))
        self.hand_status = {}
        self.mirror = bool(settings.get("mirror", True))
        self.show_overlay = bool(settings.get("show_overlay", True))
        self.fps = 0.0
        self._last = None
        self.last_features = None
        self.last_ctx = None
        self.events = []

    def set_mode(self, mode):
        if mode in MODES:
            self.mode = mode
            self.renderer.settings.performance = (mode == "performance")
            if mode in ("sandbox", "calibration"):
                self._cancel_fireball_charge()
                self.face_proximity.reset()

    def trigger(self, name, hand_label=None):
        ctx = self.last_ctx or FrameContext(640, 360)
        return self.renderer.trigger(name, ctx, hand_label)

    def set_control_mode(self, mode):
        if mode in ("simple", "classic"):
            self.control_mode = mode
            if mode != "simple":
                self._cancel_fireball_charge()
                self.face_proximity.reset()

    def set_face_enabled(self, enabled):
        self.face_enabled = bool(enabled) and self.face_tracker is not None
        if not self.face_enabled:
            self.face_proximity.reset()
            self._cancel_fireball_charge()

    def set_power_enabled(self, power, enabled):
        self.power_enabled[power] = bool(enabled)
        if power == "Fireball" and not enabled:
            self._cancel_fireball_charge()

    def reset_tracking_states(self):
        self.simple_controls.reset()
        self.face_proximity.reset()
        fireball = self.renderer.effects["fireball"]
        if not fireball.launched:
            fireball.active = False
        self.fireball_charge_label = None
        self.face_cooldown_until = 0.0
        self.hand_status = {}
    def set_face_enabled(self, enabled):
        self.face_enabled = bool(enabled) and self.face_tracker is not None
        if not self.face_enabled:
            self.face_proximity.reset()
            self._cancel_fireball_charge()

    def set_power_enabled(self, power, enabled):
        self.power_enabled[power] = bool(enabled)
        if power == "Fireball" and not enabled:
            self._cancel_fireball_charge()

    def _cancel_fireball_charge(self):
        fireball = self.renderer.effects["fireball"]
        if not fireball.launched:
            fireball.active = False
        self.fireball_charge_label = None

    def process(self, frame, now=None):
        t_start = time.perf_counter()
        now = time.monotonic() if now is None else now
        if self.mirror:
            frame = cv2.flip(frame, 1)
        pw = 480 if self.mode == "performance" else int(self.settings.get("proc_width", 640))
        h0, w0 = frame.shape[:2]
        if w0 != pw:
            frame = cv2.resize(frame, (pw, int(h0 * pw / w0)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

        t_inf = time.perf_counter()
        hands = self.tracker.process(frame, int(now * 1000)) if self.tracker else []
        face_tracking = (self.face_enabled and self.face_tracker is not None and self.control_mode == "simple"
                 and self.mode not in ("sandbox", "calibration"))
        face = self.face_tracker.process(frame, int(now * 1000)) if face_tracking else None
        infer_ms = (time.perf_counter() - t_inf) * 1000
        if not self.mirror:  # MediaPipe labels assume a mirrored image
            for hd in hands:
                hd.label = "Left" if hd.label == "Right" else "Right"

        gs = self.classifier.classify(hands, now)
        transitions, self.hand_status = self.simple_controls.update(hands, self.classifier.classify_hand, now)
        face_entries, self.face_status = self.face_proximity.update(hands, face)
        self.face_status.update(available=self.face_enabled and self.face_tracker is not None,
                    tracking=face_tracking)
        self.last_features = feature_vector(hands[0].lm) if hands else None
        dt = 0.033 if self._last is None else min(0.1, max(0.005, now - self._last))
        ctx = FrameContext(w, h, hands, dt)
        self.last_ctx = ctx

        self.events = []
        simple_active = self.simple_enabled and self.control_mode == "simple"
        if simple_active and self.mode not in ("sandbox", "calibration"):
            if self.face_enabled and self.face_tracker and self.power_enabled.get("Fireball", True):
                if self.fireball_charge_label and not self.face_status["near"].get(self.fireball_charge_label):
                    fireball = self.renderer.effects["fireball"]
                    if not fireball.launched:
                        fireball.active = False
                    self.fireball_charge_label = None
                if self.fireball_charge_label is None and now >= self.face_cooldown_until:
                    if face_entries:
                        label = face_entries[0]
                        if self.face_status["near"].get(label):
                            self.renderer.trigger("fireball", ctx, label)
                            fireball = self.renderer.effects["fireball"]
                            if self.fireball_proximity_only:
                                fireball.launch(ctx)
                                self.events.append(f"{label}: fireball launched")
                            else:
                                fireball.hold_charge()
                                self.fireball_charge_label = label
                                self.events.append(f"{label}: fireball charging")
                            self.face_cooldown_until = now + self.simple_controls.cooldown
            for transition in transitions:
                if self.fireball_charge_label == transition.label and self.power_enabled.get("Fireball", True):
                    if self.renderer.effects["fireball"].launch(ctx):
                        self.events.append(f"{transition.label}: fireball launched")
                    self.fireball_charge_label = None
                    continue
                effect = "rasengan" if transition.label == "Right" else "chidori"
                if self.power_enabled.get(transition.label, True):
                    self.renderer.trigger(effect, ctx, transition.label)
                    self.events.append(f"{transition.label}: {effect}")
            right = self.hand_status.get("Right", {})
            rasengan = self.renderer.effects["rasengan"]
            if (self.rasengan_motion_launch and right.get("present") and right.get("pose") == "open_palm"
                    and right.get("scale_velocity", 0.0) >= 90 and rasengan.active and not rasengan.launched):
                if rasengan.launch(ctx):
                    self.events.append("Right: rasengan launched")
            left = self.hand_status.get("Left", {})
            chidori = self.renderer.effects["chidori"]
            if (self.chidori_motion_discharge and left.get("present") and left.get("pose") == "open_palm"
                    and left.get("scale_velocity", 0.0) >= 90 and chidori.active):
                self.renderer.trigger("chidori", ctx, "Left")
                self.events.append("Left: chidori discharged")
        elif self.mode not in ("sandbox", "calibration"):
            for name, ev, effect in self.sequences.update(gs.name, now):
                self.events.append(f"{name}: {ev}")
                if ev == "completed":
                    self.renderer.trigger(effect, ctx)

        out = self.renderer.render(frame, ctx)

        prog = self.sequences.progress(now)
        loop = None if self._last is None else now - self._last
        if loop and loop > 0:
            self.fps = 1 / loop if self.fps == 0 else 0.9 * self.fps + 0.1 / loop
        self._last = now
        status = dict(hands=len(hands), gesture=gs.name, confidence=gs.confidence, per_hand=gs.per_hand,
                      sequence=prog, fps=self.fps, infer_ms=infer_ms, mode=self.mode,
                      latency_ms=(time.perf_counter() - t_start) * 1000, events=self.events,
                      active_effects=self.renderer.active_names(),
                      control_mode=self.control_mode,
                      hand_status=self.hand_status,
                      face_status=self.face_status,
                      calib_samples=len(self.calibration.samples) if self.calibration else 0)
        if self.mode != "cinematic" and self.show_overlay:
            overlay.draw_landmarks(out, hands, [p[1] for p in gs.per_hand] if self.mode in ("practice", "calibration") else None)
            overlay.draw_hud(out, status)
        return out, status

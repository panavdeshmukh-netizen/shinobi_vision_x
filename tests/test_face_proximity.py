import numpy as np

from vision.face_tracker import FaceProximityController, FaceRegion


class FakeHand:
    def __init__(self, label, position):
        self.label = label
        self.palm_center = np.asarray(position, dtype=float)


def test_face_proximity_is_scaled_by_face_size_and_debounced():
    gate = FaceProximityController(threshold=0.5, stable_frames=2)
    face = FaceRegion(np.array([100.0, 100.0]), 30.0, 100.0)
    hand = FakeHand("Right", (140.0, 100.0))
    events, status = gate.update([hand], face)
    assert events == []
    assert not status["near"]["Right"]
    events, status = gate.update([hand], face)
    assert events == ["Right"]
    assert status["near"]["Right"]
    events, _ = gate.update([hand], face)
    assert events == []


def test_outside_face_region_does_not_activate_and_leaving_rearms():
    gate = FaceProximityController(threshold=0.5, stable_frames=1)
    face = FaceRegion(np.array([100.0, 100.0]), 30.0, 100.0)
    near = FakeHand("Left", (110.0, 110.0))
    far = FakeHand("Left", (250.0, 100.0))
    assert gate.update([far], face)[0] == []
    assert gate.update([near], face)[0] == ["Left"]
    assert gate.update([far], face)[0] == []
    assert gate.update([near], face)[0] == ["Left"]


def test_face_tracking_loss_clears_proximity_state():
    gate = FaceProximityController(threshold=0.5, stable_frames=1)
    face = FaceRegion(np.array([100.0, 100.0]), 30.0, 100.0)
    hand = FakeHand("Right", (110.0, 110.0))
    gate.update([hand], face)
    _, status = gate.update([hand], None)
    assert not status["face_found"]
    assert not status["near"]["Right"]
    assert gate.update([hand], face)[0] == ["Right"]


def test_fireball_charge_launches_only_on_same_hand_transition():
    from core.engine import Pipeline
    from tests.helpers import fist as make_fist, open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        hands = []

        def process(self, frame, timestamp):
            return self.hands

    class FakeFaceTracker:
        def process(self, frame, timestamp):
            return FaceRegion(np.array([320.0, 150.0]), 35.0, 200.0)

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True, "face_proximity_enabled": True,
                "jutsu_cooldown": 0.65}
    tracker = FakeTracker()
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, tracker, GestureClassifier(), SequenceEngine([]), renderer,
                        face_tracker=FakeFaceTracker())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    close = (325.0, 160.0)
    tracker.hands = [make_fist(label="Right", offset=(close[0] / 640, close[1] / 360))]
    pipeline.process(frame.copy(), 0.0)
    _, status = pipeline.process(frame.copy(), 0.03)
    assert status["events"] == ["Right: fireball charging"]
    effect = renderer.effects["fireball"]
    assert effect.active and effect.manual_charge and not effect.launched

    tracker.hands = [make_fist(label="Left"), open_hand(label="Right", offset=(close[0] / 640, close[1] / 360))]
    pipeline.process(frame.copy(), 0.06)
    _, status = pipeline.process(frame.copy(), 0.09)
    assert status["events"] == ["Right: fireball launched"]
    assert effect.launched
    assert effect.hand_label == "Right"


def test_fireball_disabled_during_charge_cancels_pending_effect():
    from core.engine import Pipeline
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.base import FrameContext
    from vfx.renderer import VFXRenderer

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "face_proximity_enabled": True}
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, None, GestureClassifier(), SequenceEngine([]), renderer, face_tracker=object())
    pipeline.fireball_charge_label = "Right"
    pipeline.last_ctx = FrameContext(640, 360)
    renderer.trigger("fireball", pipeline.last_ctx, "Right")
    renderer.effects["fireball"].hold_charge()
    pipeline.set_power_enabled("Fireball", False)
    assert pipeline.fireball_charge_label is None
    assert not renderer.effects["fireball"].active


def test_proximity_only_fireball_launches_immediately_on_face_entry():
    from core.engine import Pipeline
    from tests.helpers import open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        def process(self, frame, timestamp):
            return [open_hand(label="Right", offset=(325 / 640, 160 / 360))]

    class FakeFaceTracker:
        def process(self, frame, timestamp):
            return FaceRegion(np.array([320.0, 150.0]), 35.0, 200.0)

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True, "face_proximity_enabled": True,
                "fireball_proximity_only": True}
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, FakeTracker(), GestureClassifier(), SequenceEngine([]), renderer,
                        face_tracker=FakeFaceTracker())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    pipeline.process(frame.copy(), 0.0)
    _, status = pipeline.process(frame.copy(), 0.03)
    assert status["events"] == ["Right: fireball launched"]
    assert renderer.effects["fireball"].launched
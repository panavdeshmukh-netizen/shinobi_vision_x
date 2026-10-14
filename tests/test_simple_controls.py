import numpy as np

from vision.simple_controls import SimpleGestureController


class FakeHand:
    def __init__(self, label, pose, score=1.0, position=(10.0, 20.0)):
        self.label = label
        self.pose = pose
        self.score = score
        self.palm_center = np.asarray(position, dtype=float)
        self.direction = np.array([0.0, -1.0])
        self.scale = 20.0


def classify(hand):
    return hand.pose, 0.95


def feed(controller, now, *hands):
    return controller.update(hands, classify, now)


def test_hands_activate_independently_on_fist_to_open():
    controller = SimpleGestureController(stable_frames=1)
    feed(controller, 0.0, FakeHand("Right", "fist"), FakeHand("Left", "fist"))
    events, _ = feed(controller, 0.1, FakeHand("Right", "open_palm"), FakeHand("Left", "fist"))
    assert [event.label for event in events] == ["Right"]
    events, _ = feed(controller, 0.2, FakeHand("Right", "open_palm"), FakeHand("Left", "open_palm"))
    assert [event.label for event in events] == ["Left"]


def test_open_palm_does_not_repeat_and_fist_rearms_after_cooldown():
    controller = SimpleGestureController(cooldown=0.5, stable_frames=1)
    feed(controller, 0.0, FakeHand("Right", "fist"))
    events, _ = feed(controller, 0.1, FakeHand("Right", "open_palm"))
    assert len(events) == 1
    events, _ = feed(controller, 0.7, FakeHand("Right", "open_palm"))
    assert not events
    feed(controller, 0.8, FakeHand("Right", "fist"))
    events, _ = feed(controller, 0.9, FakeHand("Right", "open_palm"))
    assert len(events) == 1


def test_tracking_loss_requires_a_new_fist_before_activation():
    controller = SimpleGestureController(stable_frames=1, loss_timeout=0.2)
    feed(controller, 0.0, FakeHand("Left", "fist"))
    feed(controller, 0.1)
    feed(controller, 0.4)
    events, state = feed(controller, 0.5, FakeHand("Left", "open_palm"))
    assert not events
    assert state["Left"]["pose"] == "open_palm"
    feed(controller, 0.6, FakeHand("Left", "fist"))
    events, _ = feed(controller, 0.7, FakeHand("Left", "open_palm"))
    assert len(events) == 1


def test_short_confidence_noise_does_not_change_pose():
    controller = SimpleGestureController(stable_frames=2)
    feed(controller, 0.0, FakeHand("Right", "fist"))
    events, _ = controller.update([FakeHand("Right", "open_palm")], lambda _: ("open_palm", 0.2), 0.1)
    assert not events
    events, _ = feed(controller, 0.2, FakeHand("Right", "fist"))
    assert not events


def test_targeted_anchor_selects_the_requested_handedness():
    from vfx.base import FrameContext

    left = FakeHand("Left", "open_palm", position=(20.0, 30.0))
    right = FakeHand("Right", "open_palm", position=(90.0, 80.0))
    ctx = FrameContext(120, 100, [left, right])
    assert np.allclose(ctx.hand_anchor(label="Right"), right.palm_center)
    assert np.allclose(ctx.hand_anchor(label="Left"), left.palm_center)
    assert ctx.hand_anchor(label="Unknown") is None


def test_pipeline_dispatches_right_hand_transition_to_rasengan():
    from core.engine import Pipeline
    from tests.helpers import fist as make_fist, open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        hands = []

        def process(self, frame, timestamp):
            return self.hands

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True}
    tracker = FakeTracker()
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, tracker, GestureClassifier(), SequenceEngine([]), renderer)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.hands = [make_fist(label="Right")]
    pipeline.process(frame.copy(), 0.0)
    pipeline.process(frame.copy(), 0.03)
    tracker.hands = [open_hand(label="Right")]
    pipeline.process(frame.copy(), 0.06)
    _, status = pipeline.process(frame.copy(), 0.09)
    assert status["events"] == ["Right: rasengan"]
    assert renderer.effects["rasengan"].hand_label == "Right"
    assert renderer.effects["rasengan"].active

def test_pipeline_dispatches_left_hand_transition_to_chidori():
    from core.engine import Pipeline
    from tests.helpers import fist as make_fist, open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        hands = []

        def process(self, frame, timestamp):
            return self.hands

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True}
    tracker = FakeTracker()
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, tracker, GestureClassifier(), SequenceEngine([]), renderer)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.hands = [make_fist(label="Left")]
    pipeline.process(frame.copy(), 0.0)
    pipeline.process(frame.copy(), 0.03)
    tracker.hands = [open_hand(label="Left")]
    pipeline.process(frame.copy(), 0.06)
    _, status = pipeline.process(frame.copy(), 0.09)
    assert status["events"] == ["Left: chidori"]
    assert renderer.effects["chidori"].hand_label == "Left"
    assert renderer.effects["chidori"].active


def test_unmirrored_preview_corrects_media_pipe_handedness_before_dispatch():
    import copy

    from core.engine import Pipeline
    from tests.helpers import fist as make_fist, open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        hands = []

        def process(self, frame, timestamp):
            return copy.deepcopy(self.hands)

    settings = {"mode": "practice", "mirror": False, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True}
    tracker = FakeTracker()
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, tracker, GestureClassifier(), SequenceEngine([]), renderer)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.hands = [make_fist(label="Left")]
    pipeline.process(frame.copy(), 0.0)
    pipeline.process(frame.copy(), 0.03)
    tracker.hands = [open_hand(label="Left")]
    pipeline.process(frame.copy(), 0.06)
    _, status = pipeline.process(frame.copy(), 0.09)
    assert status["events"] == ["Right: rasengan"]
    assert renderer.effects["rasengan"].hand_label == "Right"


def test_forward_hand_expansion_launches_an_active_rasengan_when_enabled():
    from core.engine import Pipeline
    from tests.helpers import fist as make_fist, open_hand
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceEngine
    from vfx.renderer import VFXRenderer

    class FakeTracker:
        hands = []

        def process(self, frame, timestamp):
            return self.hands

    settings = {"mode": "practice", "mirror": True, "show_overlay": False, "proc_width": 640,
                "control_mode": "simple", "simple_enabled": True, "rasengan_motion_launch": True}
    tracker = FakeTracker()
    renderer = VFXRenderer()
    pipeline = Pipeline(settings, tracker, GestureClassifier(), SequenceEngine([]), renderer)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.hands = [make_fist(label="Right")]
    pipeline.process(frame.copy(), 0.0)
    pipeline.process(frame.copy(), 0.03)
    hand = open_hand(label="Right")
    tracker.hands = [hand]
    pipeline.process(frame.copy(), 0.06)
    pipeline.process(frame.copy(), 0.09)
    assert renderer.effects["rasengan"].active
    before = float(hand.scale)
    hand.px[9] = hand.px[0] + (hand.px[9] - hand.px[0]) * 2.0
    assert hand.scale > before
    _, status = pipeline.process(frame.copy(), 0.12)
    assert status["events"] == ["Right: rasengan launched"]
    assert renderer.effects["rasengan"].launched
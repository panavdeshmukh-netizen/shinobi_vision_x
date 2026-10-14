import numpy as np
import pytest
from tests.helpers import open_hand, fist
from vfx.base import FrameContext, VFXSettings
from vfx.compositing import additive, alpha_over, bloom
from vfx.renderer import VFXRenderer

W, H = 640, 360


def ctx(hands=None):
    return FrameContext(W, H, hands if hands is not None else [open_hand(), open_hand(offset=(0.2, 0.6), label="Left")], 0.033)


def test_alpha_over_extremes():
    dst = np.full((10, 12, 3), 100, np.uint8)
    src = np.zeros((10, 12, 4), np.uint8)
    src[..., :3] = 200
    out = alpha_over(dst.copy(), src)
    assert (out == 100).all()                      # alpha 0 -> unchanged
    src[..., 3] = 255
    assert (alpha_over(dst.copy(), src) == 200).all()
    src[..., 3] = 128
    assert abs(int(alpha_over(dst.copy(), src)[0, 0, 0]) - 150) <= 1


def test_additive_saturates_and_bloom_keeps_shape():
    a = np.full((20, 20, 3), 200, np.uint8)
    additive(a, np.full_like(a, 100))
    assert (a == 255).all()
    layer = np.zeros((H, W, 3), np.uint8)
    layer[170:190, 310:330] = 255
    out = bloom(layer, 0.6)
    assert out.shape == layer.shape and out.dtype == np.uint8 and out[180, 336].max() > 0


@pytest.mark.parametrize("name", VFXRenderer().names)
def test_every_effect_renders_without_error(name):
    r = VFXRenderer(VFXSettings(color_grade=True, screen_shake=True))
    base = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    c = ctx()
    for _ in range(12):
        r.render(base.copy(), c)
    assert r.trigger(name, c)
    changed = False
    for _ in range(30):
        f = base.copy()
        out = r.render(f, c)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        changed |= bool((out != base).any())
    assert changed


def test_effects_expire_and_flash_is_capped():
    r = VFXRenderer()
    c = ctx()
    r.trigger("lightning", c)
    f = np.zeros((H, W, 3), np.uint8)
    r.render(f, c)
    assert f.mean() < 90                           # no full-frame white-out
    for _ in range(200):
        r.render(np.zeros((H, W, 3), np.uint8), c)
    assert r.active_names() == []


def test_effect_follows_hand():
    r = VFXRenderer()
    c1 = ctx([open_hand(offset=(0.3, 0.5))])
    r.trigger("rasengan", c1)
    c2 = ctx([open_hand(offset=(0.7, 0.5))])
    for _ in range(25):
        r.render(np.zeros((H, W, 3), np.uint8), c2)
    assert abs(r.effects["rasengan"].pos[0] - 0.7 * W) < 0.12 * W


def test_works_with_no_hands():
    r = VFXRenderer()
    c = ctx([])
    for n in r.names:
        r.trigger(n, c)
    for _ in range(5):
        r.render(np.zeros((H, W, 3), np.uint8), c)


def test_pipeline_with_fake_tracker_and_sequence():
    from core.config import Settings
    from core.engine import Pipeline
    from vision.gesture_classifier import GestureClassifier
    from vision.sequence_engine import SequenceDef, SequenceEngine

    class FakeTracker:           # unit-test stand-in ONLY; the app always uses the real tracker
        def __init__(self): self.hands = []
        def process(self, f, ts): return self.hands

    tr = FakeTracker()
    seq = SequenceEngine([SequenceDef("s", ["fist", "open_palm"], "lightning", 0.2, 3, 2)])
    s = Settings()
    s["mirror"] = False
    s["mode"] = "practice"
    s["control_mode"] = "classic"
    p = Pipeline(s, tr, GestureClassifier(), seq, VFXRenderer())
    frame = np.zeros((480, 854, 3), np.uint8)
    now = 0.0
    st = None
    for pose in (fist, open_hand):
        tr.hands = [pose()]
        for _ in range(14):
            out, st = p.process(frame.copy(), now)
            now += 0.05
    assert out.shape[1] == s["proc_width"]
    assert "lightning" in st["active_effects"]

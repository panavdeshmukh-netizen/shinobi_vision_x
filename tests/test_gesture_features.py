import numpy as np
from tests.helpers import make_hand, open_hand, fist
from vision.gesture_classifier import GestureClassifier
from vision.gesture_features import feature_vector, finger_states, normalized_landmarks


def cls(h):
    return GestureClassifier().classify_hand(h)[0]


def test_core_poses():
    assert cls(open_hand()) == "open_palm"
    assert cls(fist()) == "fist"
    assert cls(make_hand(thumb="fold", index=True)) == "pointing"
    assert cls(make_hand(thumb="fold", index=True, middle=True)) == "peace"
    assert cls(make_hand(thumb="up")) == "thumbs_up"


def test_finger_states():
    st = finger_states(make_hand(index=True, middle=True).lm)
    assert st["index"] and st["middle"] and not st["ring"] and not st["pinky"]


def test_invariance_to_scale_translation_rotation():
    a = open_hand(scale=0.2, rot=0.0, offset=(0.4, 0.5))
    b = open_hand(scale=0.35, rot=0.7, offset=(0.6, 0.45))
    assert np.allclose(feature_vector(a.lm), feature_vector(b.lm), atol=2e-3)
    assert cls(b) == "open_palm"
    na, nb = normalized_landmarks(a.lm), normalized_landmarks(b.lm)
    assert np.allclose(na, nb, atol=2e-3)


def test_two_hand_poses():
    c = GestureClassifier()
    far = [open_hand(offset=(0.25, 0.6), label="Left"), open_hand(offset=(0.75, 0.6), label="Right")]
    assert c.classify_frame(far)[0] == "two_hand_open"
    near = [open_hand(offset=(0.45, 0.6), label="Left"), open_hand(offset=(0.55, 0.6), label="Right")]
    assert c.classify_frame(near)[0] in ("two_hand_aligned", "two_hand_contact")
    touch = [open_hand(offset=(0.49, 0.6), label="Left"), open_hand(offset=(0.51, 0.6), label="Right")]
    assert c.classify_frame(touch)[0] == "two_hand_contact"
    assert c.classify_frame([open_hand()])[0] == "open_palm"  # one hand never triggers two-hand poses


def test_smoothing_prevents_flicker():
    c = GestureClassifier(window=6, vote=0.6)
    for _ in range(6):
        c.classify([open_hand()])
    s = c.classify([fist()])      # single noisy frame
    assert s.name == "open_palm"
    for _ in range(6):
        s = c.classify([fist()])
    assert s.name == "fist"


def test_custom_pose_knn():
    h = make_hand(thumb="out", index=True, middle=False, ring=False, pinky=True)
    c = GestureClassifier(custom={"horns": {"samples": [feature_vector(h.lm)]}})
    assert c.classify_hand(h)[0] == "horns"

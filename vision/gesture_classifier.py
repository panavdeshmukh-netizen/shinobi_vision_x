"""Rule-based pose classifier + custom kNN poses + temporal smoothing."""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .gesture_features import feature_vector, finger_scores, palm_size

BUILTIN = ("open_palm", "fist", "pointing", "peace", "thumbs_up",
           "two_hand_open", "two_hand_aligned", "two_hand_contact")


@dataclass
class GestureState:
    name: str = "none"
    raw: str = "none"
    confidence: float = 0.0
    per_hand: list = field(default_factory=list)


class GestureClassifier:
    def __init__(self, custom=None, window=6, vote=0.6, min_conf=0.5):
        self.window, self.vote, self.min_conf = window, vote, min_conf
        self.hist = deque(maxlen=window)
        self.conf_hist = deque(maxlen=window)
        self.stable = "none"
        self.custom = {}
        self.set_custom(custom or {})

    def set_custom(self, custom):
        self.custom = {n: np.array(v["samples"] if isinstance(v, dict) else v, float)
                       for n, v in custom.items() if len(v["samples"] if isinstance(v, dict) else v)}

    @property
    def pose_names(self):
        return list(BUILTIN) + sorted(self.custom)

    def classify_hand(self, hand):
        lm = np.asarray(hand.lm, float)
        sc = finger_scores(lm)
        on = {k: v > 0.5 for k, v in sc.items()}
        four = [on["index"], on["middle"], on["ring"], on["pinky"]]
        margin = float(np.clip(0.5 + np.mean([abs(v - 0.5) for v in sc.values()]), 0.3, 1.0))
        up = (lm[0][1] - lm[4][1]) / palm_size(lm)
        if on["thumb"] and not any(four) and up > 0.7:
            name = "thumbs_up"
        elif all(four):
            name = "open_palm"
        elif not any(four) and not on["thumb"]:
            name = "fist"
        elif on["index"] and not (on["middle"] or on["ring"] or on["pinky"]):
            name = "pointing"
        elif on["index"] and on["middle"] and not (on["ring"] or on["pinky"]):
            name = "peace"
        else:
            name, margin = "unknown", 0.3
        if self.custom:
            fv = np.array(feature_vector(lm))
            for cname, samples in self.custom.items():
                d = float(np.min(np.linalg.norm(samples - fv, axis=1)) / np.sqrt(len(fv)))
                cc = float(np.exp(-d / 0.08))
                if cc >= 0.7 and cc > margin * 0.9:
                    name, margin = cname, cc
        return name, margin

    def classify_frame(self, hands):
        """Return (raw_name, conf, per_hand)."""
        if not hands:
            return "none", 0.0, []
        per = [(h.label, *self.classify_hand(h)) for h in hands]
        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            ms = (a.scale + b.scale) / 2
            diff = a.palm_center - b.palm_center
            dist = float(np.linalg.norm(diff))
            both_open = per[0][1] == "open_palm" and per[1][1] == "open_palm"
            conf = min(per[0][2], per[1][2])
            if dist < 1.3 * ms:
                return "two_hand_contact", max(conf, 0.6), per
            if both_open and dist < 2.6 * ms and abs(diff[1]) < 0.8 * ms:
                return "two_hand_aligned", conf, per
            if both_open:
                return "two_hand_open", conf, per
        best = max(zip(hands, per), key=lambda x: x[0].score)[1]
        return best[1], best[2], per

    def classify(self, hands, now=None):
        raw, conf, per = self.classify_frame(hands)
        if conf < self.min_conf and raw not in ("none",):
            raw = "unknown"
        self.hist.append(raw)
        self.conf_hist.append(conf)
        name, cnt = Counter(self.hist).most_common(1)[0]
        if cnt / len(self.hist) >= self.vote:
            self.stable = name
        if not hands and list(self.hist)[-3:] == ["none"] * 3:
            self.stable = "none"
        return GestureState(self.stable, raw, float(np.mean(self.conf_hist)), per)

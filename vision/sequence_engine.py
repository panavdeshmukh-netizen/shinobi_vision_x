"""Ordered multi-pose sequence state machines (clock is passed in => fully testable)."""
from dataclasses import dataclass

IGNORED = ("none", "unknown", "", None)


@dataclass
class SequenceDef:
    name: str
    steps: list
    effect: str = "lightning"
    hold_time: float = 0.35
    timeout: float = 4.0
    cooldown: float = 3.0

    @staticmethod
    def from_dict(d):
        return SequenceDef(d["name"], list(d["steps"]), d.get("effect", "lightning"),
                           float(d.get("hold_time", 0.35)), float(d.get("timeout", 4.0)),
                           float(d.get("cooldown", 3.0)))

    def to_dict(self):
        return dict(name=self.name, steps=self.steps, effect=self.effect, hold_time=self.hold_time,
                    timeout=self.timeout, cooldown=self.cooldown)


class SequenceTracker:
    def __init__(self, d):
        self.d = d
        self.reset()
        self.cooldown_until = float("-inf")

    def reset(self):
        self.idx = 0
        self.hold_start = None
        self.last_progress = None
        self.release_pose = None  # pose just completed; ignore until user changes pose

    def cooldown_left(self, now):
        return max(0.0, self.cooldown_until - now)

    def hold_fraction(self, now):
        if self.hold_start is None:
            return 0.0
        return min(1.0, (now - self.hold_start) / max(self.d.hold_time, 1e-6))

    def update(self, pose, now):
        """Returns None | 'advanced' | 'completed' | 'reset_wrong' | 'reset_timeout'."""
        if now < self.cooldown_until:
            return None
        ev = None
        if self.idx > 0 and now - self.last_progress > self.d.timeout:
            self.reset()
            ev = "reset_timeout"
        if self.release_pose is not None:
            if pose != self.release_pose:
                self.release_pose = None
            else:
                return ev
        expected = self.d.steps[self.idx]
        if pose == expected:
            if self.hold_start is None:
                self.hold_start = now
            if now - self.hold_start >= self.d.hold_time:
                self.idx += 1
                self.last_progress = now
                self.hold_start = None
                self.release_pose = pose
                if self.idx >= len(self.d.steps):
                    self.reset()
                    self.cooldown_until = now + self.d.cooldown
                    return "completed"
                return "advanced"
        else:
            self.hold_start = None
            if self.idx > 0 and pose not in IGNORED:
                self.reset()
                return "reset_wrong"
        return ev


class SequenceEngine:
    def __init__(self, defs=()):
        self.reload(defs)

    def reload(self, defs):
        self.trackers = [SequenceTracker(d if isinstance(d, SequenceDef) else SequenceDef.from_dict(d)) for d in defs]

    def update(self, pose, now):
        """Returns list of (sequence_name, event, effect)."""
        out = []
        for t in self.trackers:
            ev = t.update(pose, now)
            if ev:
                out.append((t.d.name, ev, t.d.effect))
        return out

    def progress(self, now):
        """Most advanced tracker: dict(name, done, total, hold, cooldown)."""
        best = max(self.trackers, key=lambda t: (t.idx, t.hold_fraction(now)), default=None)
        if best is None:
            return dict(name="-", done=0, total=0, hold=0.0, cooldown=0.0)
        cd = max((t.cooldown_left(now) for t in self.trackers), default=0.0)
        return dict(name=best.d.name, done=best.idx, total=len(best.d.steps),
                    hold=best.hold_fraction(now), cooldown=cd)

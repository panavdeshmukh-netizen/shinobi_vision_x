"""Low-latency, independent gesture transitions for each hand."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandTransition:
    label: str
    previous_pose: str
    pose: str
    position: np.ndarray
    velocity: np.ndarray
    confidence: float


@dataclass
class _HandState:
    pose: str = "none"
    previous_pose: str = "none"
    pending_pose: str = "none"
    pending_count: int = 0
    confidence: float = 0.0
    position: np.ndarray | None = None
    previous_position: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    scale: float | None = None
    scale_velocity: float = 0.0
    last_seen: float | None = None
    cooldown_until: float = 0.0
    armed: bool = False


class SimpleGestureController:
    """Detect fist-to-open transitions without coupling left and right hands."""

    LABELS = ("Left", "Right")

    def __init__(self, cooldown=0.8, loss_timeout=0.3, stable_frames=2, min_confidence=0.45):
        self.cooldown = float(cooldown)
        self.loss_timeout = float(loss_timeout)
        self.stable_frames = max(1, int(stable_frames))
        self.min_confidence = float(min_confidence)
        self.states = {label: _HandState() for label in self.LABELS}

    def reset(self):
        self.states = {label: _HandState() for label in self.LABELS}

    def update(self, hands, classify_hand, now):
        """Return transitions and serializable live status, keyed by handedness."""
        now = float(now)
        observed = {}
        for hand in hands:
            if hand.label in self.states and (hand.label not in observed or hand.score > observed[hand.label].score):
                observed[hand.label] = hand

        events = []
        status = {}
        for label, state in self.states.items():
            hand = observed.get(label)
            if hand is None:
                if state.last_seen is not None and now - state.last_seen >= self.loss_timeout:
                    state.pose = state.previous_pose = "none"
                    state.pending_pose, state.pending_count = "none", 0
                    state.position = state.previous_position = None
                    state.velocity = np.zeros(2, dtype=float)
                    state.scale = None
                    state.scale_velocity = 0.0
                    state.confidence = 0.0
                    state.last_seen = None
                    state.armed = False
                status[label] = self._status(state, now, False)
                continue

            if state.last_seen is not None and now - state.last_seen >= self.loss_timeout:
                state.pose = state.previous_pose = "none"
                state.pending_pose, state.pending_count = "none", 0
                state.position = state.previous_position = None
                state.velocity = np.zeros(2, dtype=float)
                state.scale = None
                state.scale_velocity = 0.0
                state.armed = False

            pose, confidence = classify_hand(hand)
            confidence = float(confidence)
            pos = np.asarray(hand.palm_center, dtype=float)
            current_scale = float(hand.scale)
            if state.position is not None and state.last_seen is not None and now > state.last_seen:
                state.velocity = (pos - state.position) / (now - state.last_seen)
                if state.scale is not None:
                    measured = (current_scale - state.scale) / (now - state.last_seen)
                    state.scale_velocity = 0.5 * state.scale_velocity + 0.5 * measured
            else:
                state.velocity = np.zeros(2, dtype=float)
                state.scale_velocity = 0.0
            state.previous_position = state.position
            state.position = pos.copy()
            state.scale = current_scale
            state.last_seen = now
            state.confidence = confidence

            if confidence >= self.min_confidence and pose in ("fist", "open_palm"):
                if pose == state.pending_pose:
                    state.pending_count += 1
                else:
                    state.pending_pose, state.pending_count = pose, 1
                if state.pending_count >= self.stable_frames and pose != state.pose:
                    previous = state.pose
                    state.previous_pose, state.pose = previous, pose
                    if pose == "fist":
                        state.armed = True
                    elif previous == "fist" and state.armed and now >= state.cooldown_until:
                        events.append(HandTransition(
                            label=label,
                            previous_pose=previous,
                            pose=pose,
                            position=pos.copy(),
                            velocity=state.velocity.copy(),
                            confidence=confidence,
                        ))
                        state.cooldown_until = now + self.cooldown
                        state.armed = False
            else:
                state.pending_pose, state.pending_count = "none", 0

            status[label] = self._status(state, now, True)
        return events, status

    @staticmethod
    def _status(state, now, present):
        return {
            "present": present,
            "pose": state.pose,
            "previous_pose": state.previous_pose,
            "confidence": state.confidence,
            "position": None if state.position is None else state.position.copy(),
            "velocity": state.velocity.copy(),
            "scale_velocity": state.scale_velocity,
            "cooldown_remaining": max(0.0, state.cooldown_until - now),
            "armed": state.armed,
        }
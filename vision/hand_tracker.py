"""MediaPipe hand landmark wrapper (Tasks API, with legacy solutions fallback)."""
from pathlib import Path

import cv2
import numpy as np

from .types import HandData

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


class ModelMissingError(RuntimeError):
    pass


class HandTracker:
    def __init__(self, model_path, det_conf=0.6, track_conf=0.5, num_hands=2):
        import mediapipe as mp
        self.mp = mp
        self._last_ts = -1
        self._legacy = None
        self._lm = None
        p = Path(model_path)
        if p.is_file():
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(p)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=det_conf,
                min_hand_presence_confidence=det_conf,
                min_tracking_confidence=track_conf)
            self._lm = vision.HandLandmarker.create_from_options(opts)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy = mp.solutions.hands.Hands(
                max_num_hands=num_hands, min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf)
        else:
            raise ModelMissingError(
                f"Hand model not found at {p}. Run:  python scripts/download_model.py\n"
                f"(official source: {MODEL_URL})")

    def process(self, bgr, ts_ms):
        """Returns list[HandData]. Labels are as seen by the user IF the frame is mirrored."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = []
        if self._lm is not None:
            ts_ms = max(int(ts_ms), self._last_ts + 1)
            self._last_ts = ts_ms
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self._lm.detect_for_video(img, ts_ms)
            for pts, hd in zip(res.hand_landmarks, res.handedness):
                lm = np.array([[q.x, q.y, q.z] for q in pts], dtype=np.float32)
                hands.append(HandData(hd[0].category_name, lm, lm[:, :2] * [w, h], float(hd[0].score)))
        else:
            rgb.flags.writeable = False
            res = self._legacy.process(rgb)
            if res.multi_hand_landmarks:
                for pts, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    lm = np.array([[q.x, q.y, q.z] for q in pts.landmark], dtype=np.float32)
                    c = hd.classification[0]
                    hands.append(HandData(c.label, lm, lm[:, :2] * [w, h], float(c.score)))
        return hands

    def close(self):
        if self._lm is not None:
            self._lm.close()
        if self._legacy is not None:
            self._legacy.close()

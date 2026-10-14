"""Offline face landmarks and normalized mouth-proximity gating."""
from dataclasses import dataclass

import cv2
import numpy as np


FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


@dataclass
class FaceRegion:
    mouth_center: np.ndarray
    mouth_width: float
    face_width: float


class FaceTracker:
    def __init__(self, model_path):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        import mediapipe as mp
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp = -1

    def process(self, bgr, timestamp_ms):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(int(timestamp_ms), self.last_timestamp + 1)
        self.last_timestamp = timestamp_ms
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        points = result.face_landmarks[0]
        xy = np.asarray([(point.x * w, point.y * h) for point in points], dtype=float)
        mouth_center = (xy[13] + xy[14]) * 0.5
        mouth_width = float(np.linalg.norm(xy[61] - xy[291]))
        face_width = float(np.linalg.norm(xy[234] - xy[454]))
        if mouth_width <= 1 or face_width <= 1:
            return None
        return FaceRegion(mouth_center, mouth_width, face_width)

    def close(self):
        self.landmarker.close()


class FaceProximityController:
    """Debounce hand entry into the lower-face region using face-relative units."""

    def __init__(self, threshold=0.68, stable_frames=2):
        self.threshold = float(threshold)
        self.stable_frames = max(1, int(stable_frames))
        self.counts = {"Left": 0, "Right": 0}
        self.near = {"Left": False, "Right": False}
        self.distance = {"Left": None, "Right": None}

    def reset(self):
        self.counts = {"Left": 0, "Right": 0}
        self.near = {"Left": False, "Right": False}
        self.distance = {"Left": None, "Right": None}

    def update(self, hands, face):
        observed = {hand.label: hand for hand in hands if hand.label in self.near}
        entered = []
        for label in self.near:
            hand = observed.get(label)
            if face is None or hand is None or face.face_width <= 1:
                self.counts[label] = 0
                self.near[label] = False
                self.distance[label] = None
                continue
            ratio = float(np.linalg.norm(hand.palm_center - face.mouth_center) / face.face_width)
            self.distance[label] = ratio
            if ratio <= self.threshold:
                self.counts[label] += 1
                if self.counts[label] >= self.stable_frames and not self.near[label]:
                    self.near[label] = True
                    entered.append(label)
            else:
                self.counts[label] = 0
                self.near[label] = False
        return entered, {
            "face_found": face is not None,
            "near": dict(self.near),
            "distance": dict(self.distance),
        }
"""Processing thread: owns camera, tracker and pipeline. GUI talks to it via signals + post()."""
import logging
import queue
import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.config import CAPTURE_DIR, CONFIG_DIR, ROOT
from core.engine import Pipeline
from vision.calibration import CalibrationSession, load_custom
from vision.camera import Camera, CameraError
from vision.gesture_classifier import GestureClassifier
from vision.hand_tracker import HandTracker
from vision.face_tracker import FaceTracker
from vision.sequence_engine import SequenceEngine
from vfx.base import VFXSettings
from vfx.renderer import VFXRenderer
from core.config import load_json

log = logging.getLogger("shinobi")


def build_pipeline(settings, tracker, face_tracker=None):
    vs = VFXSettings(settings["intensity"], settings["particle_density"], settings["glow"], settings["trail"],
                     settings["screen_shake"], settings["color_grade"])
    seqs = (load_json(CONFIG_DIR / "sequences.json", {"sequences": []}) or {}).get("sequences", [])
    clf = GestureClassifier(load_custom(CONFIG_DIR / "gestures.json"))
    return Pipeline(settings, tracker, clf, SequenceEngine(seqs), VFXRenderer(vs),
                    CalibrationSession(CONFIG_DIR / "gestures.json"), face_tracker)


class VisionWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(dict)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.cmds = queue.SimpleQueue()
        self._run = True
        self.recording = False
        self._snap = False

    def post(self, fn):
        self.cmds.put(fn)

    def snapshot(self):
        self._snap = True

    def stop(self):
        self._run = False

    def run(self):
        s = self.settings
        cam = Camera(s["camera_index"], s["camera_width"], s["camera_height"])
        tracker = None
        face_tracker = None
        writer = None
        try:
            try:
                cam.start()
            except CameraError as e:
                self.failed.emit(str(e))
                return
            try:
                tracker = HandTracker(ROOT / s["model_path"], s["min_detection_conf"], s["min_tracking_conf"])
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"Hand tracker could not start: {e}")
                return
            face_model_path = ROOT / s.get("face_model_path", "assets/models/face_landmarker.task")
            if face_model_path.is_file():
                try:
                    face_tracker = FaceTracker(face_model_path)
                except Exception as e:  # noqa: BLE001
                    if s.get("face_proximity_enabled", True):
                        self.message.emit(f"Face proximity unavailable (model initialization failed): {e}")
            elif s.get("face_proximity_enabled", True):
                self.message.emit("Face proximity unavailable; run python scripts\\download_model.py --face")
            pipe = build_pipeline(s, tracker, face_tracker)
            last_seq, last_err = -1, 0.0
            while self._run:
                while True:
                    try:
                        fn = self.cmds.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        fn(pipe)
                    except Exception:  # noqa: BLE001
                        log.exception("command failed")
                item = cam.read_latest(last_seq)
                if item is None:
                    if cam.error:
                        self.failed.emit(cam.error)
                        break
                    self.msleep(3)
                    continue
                frame, last_seq = item
                try:
                    out, st = pipe.process(frame)
                except Exception as e:  # noqa: BLE001
                    log.exception("frame failed")
                    if time.time() - last_err > 2:
                        self.message.emit(f"Processing error: {e}")
                        last_err = time.time()
                    continue
                if self.recording:
                    if writer is None:
                        CAPTURE_DIR.mkdir(exist_ok=True)
                        h, w = out.shape[:2]
                        path = CAPTURE_DIR / time.strftime("rec_%Y%m%d_%H%M%S.mp4")
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                        self.message.emit(f"Recording to {path.name}")
                    writer.write(out)
                    cv2.circle(out, (out.shape[1] - 20, 20), 7, (0, 0, 255), -1)
                    cv2.putText(out, "REC", (out.shape[1] - 62, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                elif writer is not None:
                    writer.release()
                    writer = None
                if self._snap:
                    self._snap = False
                    CAPTURE_DIR.mkdir(exist_ok=True)
                    p = CAPTURE_DIR / time.strftime("shot_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(str(p), out)
                    self.message.emit(f"Screenshot saved: {p.name}")
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                self.frame_ready.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
                self.status.emit(st)
        finally:
            if writer is not None:
                writer.release()
            if tracker is not None:
                tracker.close()
            if face_tracker is not None:
                face_tracker.close()
            cam.stop()

"""Threaded webcam capture that always keeps only the latest frame."""
import os
import threading
import time

import cv2


class CameraError(RuntimeError):
    pass


def _backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def list_cameras(max_index=4):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.index, self.width, self.height = index, width, height
        self.cap = None
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self.error = None

    def start(self):
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Cannot open camera {self.index}. It may be in use by another app, "
                "disconnected, or blocked in Windows Settings > Privacy > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise CameraError(f"Camera {self.index} opened but returned no frames (permission or driver issue).")
        self.cap = cap
        self._run = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self):
        fails = 0
        while self._run:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 60:
                    self.error = "Camera stopped delivering frames."
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read_latest(self, last_seq=-1):
        """Return (frame, seq) if there is a frame newer than last_seq else None."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None
            return self._frame, self._seq

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

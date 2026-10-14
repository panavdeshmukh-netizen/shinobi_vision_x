import importlib
import os
import sys

from .config import CONFIG_DIR, CAPTURE_DIR, ROOT, Settings


def run_checks(check_camera=True):
    res = []
    v = sys.version_info
    res.append(("Python 3.11/3.12", v[:2] in ((3, 11), (3, 12)), f"{v.major}.{v.minor}.{v.micro}"))
    for mod in ("numpy", "cv2", "mediapipe", "PySide6", "psutil"):
        try:
            m = importlib.import_module(mod)
            res.append((f"import {mod}", True, getattr(m, "__version__", "ok")))
        except Exception as e:  # noqa: BLE001
            res.append((f"import {mod}", False, str(e)[:80]))
    s = Settings()
    model = ROOT / s.get("model_path", "assets/models/hand_landmarker.task")
    res.append(("hand model file", model.is_file(), str(model) if model.is_file() else
                "missing -> run: python scripts/download_model.py"))
    for d in (CONFIG_DIR, CAPTURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
        res.append((f"writable {d.name}/", os.access(d, os.W_OK), str(d)))
    if check_camera:
        try:
            from vision.camera import list_cameras
            cams = list_cameras(3)
            res.append(("camera available", bool(cams), f"indices {cams}" if cams else "none found / in use / blocked"))
        except Exception as e:  # noqa: BLE001
            res.append(("camera available", False, str(e)[:80]))
    return res


def print_report():
    ok_all = True
    for name, ok, detail in run_checks():
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
        ok_all &= ok
    return ok_all

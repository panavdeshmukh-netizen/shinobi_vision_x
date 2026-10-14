"""Download official MediaPipe hand and optional face landmark models."""
import argparse
import sys
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")
dest = Path(__file__).resolve().parent.parent / "assets" / "models" / "hand_landmarker.task"
dest.parent.mkdir(parents=True, exist_ok=True)
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")

parser = argparse.ArgumentParser()
parser.add_argument("--face", action="store_true", help="also download the optional face landmark model")
args = parser.parse_args()
models = [(URL, dest)]
if args.face:
    models.append((FACE_URL, dest.with_name("face_landmarker.task")))

for url, path in models:
    try:
        print("Downloading", url)
        urllib.request.urlretrieve(url, path)
        print("Saved to", path)
    except Exception as e:  # noqa: BLE001
        print("Download failed:", e)
        print("Download the file manually from the URL above and place it at:", path)
        sys.exit(1)

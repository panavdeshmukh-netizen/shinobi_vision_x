"""JSON settings / gesture / sequence persistence."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("SHINOBI_CONFIG_DIR", ROOT / "config"))
CAPTURE_DIR = ROOT / "captures"


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return copy.deepcopy(default)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class Settings:
    """User settings = default_settings.json overlaid by settings.json."""

    def __init__(self, config_dir=None):
        self.dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.user_path = self.dir / "settings.json"
        self.data = {}
        self.load()

    def load(self):
        base = load_json(ROOT / "config" / "default_settings.json", {}) or {}
        user = load_json(self.user_path, {}) or {}
        self.data = {**base, **user}

    def save(self):
        save_json(self.user_path, self.data)

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)

    def __setitem__(self, k, v):
        self.data[k] = v

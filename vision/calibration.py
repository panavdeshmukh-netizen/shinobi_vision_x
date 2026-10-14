"""Capture custom poses and persist them to JSON."""
from core.config import load_json, save_json


class CalibrationSession:
    def __init__(self, path):
        self.path = path
        self.name = ""
        self.samples = []

    def start(self, name):
        self.name = name.strip().lower().replace(" ", "_")
        self.samples = []

    def add_sample(self, features):
        if not self.name or features is None:
            return False
        self.samples.append([float(x) for x in features])
        return True

    def save(self, min_samples=5):
        if not self.name or len(self.samples) < min_samples:
            return False
        data = load_json(self.path, {"custom": {}}) or {"custom": {}}
        data.setdefault("custom", {})[self.name] = {"samples": self.samples}
        save_json(self.path, data)
        return True


def load_custom(path):
    return (load_json(path, {"custom": {}}) or {}).get("custom", {})


def delete_custom(path, name):
    data = load_json(path, {"custom": {}}) or {"custom": {}}
    data.get("custom", {}).pop(name, None)
    save_json(path, data)

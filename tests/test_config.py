import json
import pytest
from core.config import Settings, load_json, save_json
from vision.calibration import CalibrationSession, load_custom
from vision.camera import Camera, CameraError


def test_settings_roundtrip(tmp_path):
    s = Settings(tmp_path)
    assert s["proc_width"] == 640           # from defaults
    s["proc_width"] = 480
    s["glow"] = 0.9
    s.save()
    s2 = Settings(tmp_path)
    assert s2["proc_width"] == 480 and s2["glow"] == 0.9 and "mirror" in s2.data


def test_corrupt_json_falls_back(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    assert load_json(p, {"a": 1}) == {"a": 1}
    assert load_json(tmp_path / "missing.json", []) == []


def test_calibration_saves_custom_pose(tmp_path):
    p = tmp_path / "gestures.json"
    c = CalibrationSession(p)
    c.start("Tiger Seal")
    for i in range(6):
        assert c.add_sample([0.1 * i] * 24)
    assert c.save()
    assert "tiger_seal" in load_custom(p)
    c2 = CalibrationSession(p)
    c2.start("x")
    c2.add_sample([0] * 24)
    assert not c2.save()                      # too few samples


def test_default_sequences_valid():
    from core.config import ROOT
    from vision.gesture_classifier import BUILTIN
    from vision.sequence_engine import SequenceDef
    data = json.loads((ROOT / "config" / "sequences.json").read_text())
    for d in data["sequences"]:
        sd = SequenceDef.from_dict(d)
        assert sd.steps and all(s in BUILTIN for s in sd.steps)


def test_missing_camera_raises_clean_error():
    with pytest.raises(CameraError):
        Camera(index=97).start()

"""Scale/translation/rotation-robust hand features from 21 landmarks (pure NumPy)."""
import numpy as np

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def angle(a, b, c):
    """Angle in degrees at b between a-b and c-b."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def palm_size(lm):
    return max(float(np.linalg.norm(lm[9] - lm[0])), 1e-6)


def finger_scores(lm):
    """Extension score per finger in [0,1] (1 = fully extended)."""
    lm = np.asarray(lm, float)
    ps = palm_size(lm)
    out = {}
    for name, (mcp, pip, _dip, tip) in FINGERS.items():
        out[name] = float(np.clip((angle(lm[mcp], lm[pip], lm[tip]) - 100) / 70, 0, 1))
    s_ang = np.clip((angle(lm[2], lm[3], lm[4]) - 120) / 50, 0, 1)
    s_dst = np.clip((np.linalg.norm(lm[4] - lm[5]) / ps - 0.4) / 0.4, 0, 1)
    out["thumb"] = float(min(s_ang, s_dst))
    return out


def finger_states(lm, thr=0.5):
    return {k: v > thr for k, v in finger_scores(lm).items()}


def normalized_landmarks(lm):
    """Translate wrist to origin, scale by palm size, rotate so wrist->middle MCP points up."""
    lm = np.asarray(lm, float)
    p = (lm - lm[0]) / palm_size(lm)
    ang = np.arctan2(p[9, 0], -p[9, 1])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, s], [-s, c]])
    out = p.copy()
    out[:, :2] = p[:, :2] @ rot.T
    return out


def feature_vector(lm):
    """24-D rotation/scale/translation invariant vector (joint angles + relative distances)."""
    lm = np.asarray(lm, float)
    ps = palm_size(lm)
    ang = [angle(lm[0], lm[1], lm[2]), angle(lm[1], lm[2], lm[3]), angle(lm[2], lm[3], lm[4])]
    for mcp, pip, dip, tip in FINGERS.values():
        ang += [angle(lm[0], lm[mcp], lm[pip]), angle(lm[mcp], lm[pip], lm[dip]), angle(lm[pip], lm[dip], lm[tip])]
    tips = [4, 8, 12, 16, 20]
    d = [np.linalg.norm(lm[t] - lm[0]) / ps / 3 for t in tips]
    d += [np.linalg.norm(lm[tips[i]] - lm[tips[i + 1]]) / ps for i in range(4)]
    return np.array(np.array(ang) / 180.0).tolist() + [float(x) for x in d]


def palm_normal_z(lm):
    """Sign of the palm normal's z (camera axis). Meaning depends on handedness; used as a raw feature."""
    lm = np.asarray(lm, float)
    n = np.cross(lm[5] - lm[0], lm[17] - lm[0])
    return float(n[2] if len(n) > 2 else 0.0)

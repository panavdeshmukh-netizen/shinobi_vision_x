"""Diagnostic overlays (kept separate from cinematic VFX rendering)."""
import cv2

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_landmarks(frame, hands, labels=None):
    for i, h in enumerate(hands):
        pts = h.px.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (80, 220, 60), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (0, 150, 255), -1, cv2.LINE_AA)
        if labels:
            x, y = pts[0]
            cv2.putText(frame, f"{h.label}: {labels[i]}", (int(x) - 40, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _text(frame, txt, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(frame, status):
    h = frame.shape[0]
    _text(frame, f"POSE {status['gesture']}  conf {status['confidence']:.2f}", (10, 22))
    _text(frame, f"FPS {status['fps']:.0f}  infer {status['infer_ms']:.0f} ms  hands {status['hands']}", (10, 44))
    p = status["sequence"]
    if p["total"] > 0 and (p["done"] > 0 or status["mode"] == "sequence"):
        _text(frame, f"{p['name']}: {p['done']} / {p['total']}", (10, h - 14), 0.7, (60, 220, 255))
    if p["cooldown"] > 0:
        _text(frame, f"cooldown {p['cooldown']:.1f}s", (10, h - 40), 0.5, (120, 120, 255))

"""Fixed-capacity vectorised particle pool."""
import cv2
import numpy as np


class ParticleSystem:
    def __init__(self, cap=300, seed=None):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((cap, 2), np.float32)
        self.vel = np.zeros((cap, 2), np.float32)
        self.life = np.zeros(cap, np.float32)
        self.maxlife = np.ones(cap, np.float32)
        self.size = np.ones(cap, np.float32)
        self.color = np.zeros((cap, 3), np.float32)

    @property
    def count(self):
        return int((self.life > 0).sum())

    def clear(self):
        self.life[:] = 0

    def emit(self, n, pos, speed=(20, 80), angle=(0, 6.283), life=(0.4, 1.0), size=(1, 3), color=(255, 200, 80), spread=0.0):
        free = np.flatnonzero(self.life <= 0)[:max(0, int(n))]
        m = len(free)
        if m == 0:
            return
        r = self.rng
        a = r.uniform(angle[0], angle[1], m)
        sp = r.uniform(speed[0], speed[1], m)
        self.pos[free] = np.asarray(pos, np.float32) + r.normal(0, spread, (m, 2)) if spread else np.asarray(pos, np.float32)
        self.vel[free] = np.stack([np.cos(a) * sp, np.sin(a) * sp], 1)
        L = r.uniform(life[0], life[1], m)
        self.life[free], self.maxlife[free] = L, L
        self.size[free] = r.uniform(size[0], size[1], m)
        self.color[free] = color

    def update(self, dt, gravity=(0, 0), drag=0.0, attract=None, k=0.0):
        alive = self.life > 0
        if not alive.any():
            return
        self.vel[alive] += np.asarray(gravity, np.float32) * dt
        if drag:
            self.vel[alive] *= max(0.0, 1 - drag * dt)
        if attract is not None and k:
            d = np.asarray(attract, np.float32) - self.pos[alive]
            self.vel[alive] += d * k * dt
        self.pos[alive] += self.vel[alive] * dt
        self.life[alive] -= dt

    def draw(self, canvas, boost=1.0):
        idx = np.flatnonzero(self.life > 0)
        if idx.size == 0:
            return
        a = np.clip(self.life[idx] / self.maxlife[idx], 0, 1)[:, None]
        cols = np.clip(self.color[idx] * a * boost, 0, 255).astype(int)
        h, w = canvas.shape[:2]
        for i, c in zip(idx, cols):
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), max(1, int(self.size[i])), (int(c[0]), int(c[1]), int(c[2])), -1)

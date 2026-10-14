from dataclasses import dataclass
import numpy as np


@dataclass
class HandData:
    label: str            # "Left" / "Right" as seen by the user
    lm: np.ndarray        # (21, 3) normalized landmarks
    px: np.ndarray        # (21, 2) pixel coords in the processed frame
    score: float = 1.0

    @property
    def palm_center(self):
        return self.px[[0, 5, 9, 13, 17]].mean(axis=0)

    @property
    def scale(self):
        return float(np.linalg.norm(self.px[9] - self.px[0])) + 1e-6

    @property
    def direction(self):
        v = self.px[12] - self.px[0]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([0.0, -1.0])

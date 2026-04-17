from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
try:
    from filterpy.kalman import KalmanFilter  # type: ignore
except Exception:  # pragma: no cover
    KalmanFilter = None  # type: ignore[misc,assignment]


@dataclass
class _KF2D:
    kf: "KalmanFilter"


class TrackKalmanSmoother:
    """
    DeepSORT already uses a Kalman filter for motion, but this optional layer smooths
    rendered positions (center points) to reduce jitter in UI/aiming overlays.
    """

    def __init__(self) -> None:
        if KalmanFilter is None:
            raise RuntimeError(
                "filterpy is not installed. Install dependencies with: pip install -r requirements.txt "
                "or run with --no-kalman-smoothing."
            )
        self._by_id: Dict[int, _KF2D] = {}

    def reset_missing(self, live_ids: set[int]) -> None:
        for tid in list(self._by_id.keys()):
            if tid not in live_ids:
                self._by_id.pop(tid, None)

    def update(self, track_id: int, cx: float, cy: float) -> Tuple[float, float]:
        if track_id not in self._by_id:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            # state: [x, y, vx, vy]
            dt = 1.0
            kf.F = np.array(
                [
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            kf.P *= 100.0
            kf.R *= 5.0
            kf.Q *= 0.1
            kf.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
            self._by_id[track_id] = _KF2D(kf=kf)

        kf = self._by_id[track_id].kf
        kf.predict()
        kf.update(np.array([[cx], [cy]], dtype=float))
        x, y = float(kf.x[0, 0]), float(kf.x[1, 0])
        return x, y


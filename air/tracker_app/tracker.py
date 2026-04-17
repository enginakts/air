from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detector import Detection
from .kalman_smoother import TrackKalmanSmoother


@dataclass(frozen=True)
class Track:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int
    cx: float
    cy: float


class DeepSortTracker:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        use_kalman_smoothing: bool = True,
    ) -> None:
        self.ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )
        self.smoother = TrackKalmanSmoother() if use_kalman_smoothing else None

    def update(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> List[Track]:
        # deep-sort-realtime expects: [ [x1,y1,x2,y2], conf, class ] list
        ds_in = []
        for d in detections:
            ds_in.append([list(d.to_xyxy()), float(d.conf), int(d.cls)])

        tracks = self.ds.update_tracks(ds_in, frame=frame_bgr)

        out: List[Track] = []
        live_ids: Set[int] = set()
        for t in tracks:
            if not t.is_confirmed():
                continue
            if t.time_since_update > 0:
                continue
            tid = int(t.track_id)
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            conf = float(getattr(t, "det_conf", 0.0) or 0.0)
            cls = int(getattr(t, "det_class", -1) or -1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            live_ids.add(tid)

            if self.smoother is not None:
                cx, cy = self.smoother.update(tid, cx, cy)

            out.append(
                Track(
                    track_id=tid,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=conf,
                    cls=cls,
                    cx=cx,
                    cy=cy,
                )
            )

        if self.smoother is not None:
            self.smoother.reset_missing(live_ids)

        return out


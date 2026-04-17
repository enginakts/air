from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    # xyxy pixel coords
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class YoloDetector:
    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        img_size: int = 960,
        conf: float = 0.35,
        iou: float = 0.45,
    ) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.img_size = img_size
        self.conf = conf
        self.iou = iou

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        # Ultralytics expects BGR ok; internally converts as needed.
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        boxes = r0.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        dets: List[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        return dets


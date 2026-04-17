from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCandidate:
    index: int
    size: Tuple[int, int]


def _try_open(idx: int) -> Optional[CameraCandidate]:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    h, w = frame.shape[:2]
    cap.release()
    return CameraCandidate(index=idx, size=(w, h))


def list_cameras(max_index: int = 10) -> List[CameraCandidate]:
    cams: List[CameraCandidate] = []
    for i in range(max_index + 1):
        c = _try_open(i)
        if c is not None:
            cams.append(c)
    return cams


def pick_camera_interactive(max_index: int = 10, window: str = "Select Camera") -> int:
    cams = list_cameras(max_index=max_index)
    if not cams:
        raise RuntimeError("No cameras found.")

    # simple UI: show list, user presses number key
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        img = np.zeros((420, 900, 3), dtype=np.uint8)
        cv2.putText(img, "Select camera index (press key). ESC to cancel.", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        y = 90
        for c in cams:
            cv2.putText(img, f"[{c.index}]  {c.size[0]}x{c.size[1]}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 180), 2, cv2.LINE_AA)
            y += 44

        cv2.imshow(window, img)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window)
            raise RuntimeError("Camera selection cancelled.")
        if ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            for c in cams:
                if c.index == idx:
                    cv2.destroyWindow(window)
                    return idx


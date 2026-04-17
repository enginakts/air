from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class HsvColor:
    h: int  # 0..179 (OpenCV HSV)
    s: int  # 0..255
    v: int  # 0..255


@dataclass
class HsvTol:
    dh: int = 10  # hue tolerance
    ds: int = 60
    dv: int = 60


def bgr_to_hsv(bgr: Tuple[int, int, int]) -> HsvColor:
    arr = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0]
    return HsvColor(int(hsv[0]), int(hsv[1]), int(hsv[2]))


def _mask_for_color(hsv_img: np.ndarray, color: HsvColor, tol: HsvTol) -> np.ndarray:
    # Hue wraps around at 180 in OpenCV.
    h0, s0, v0 = color.h, color.s, color.v
    dh, ds, dv = tol.dh, tol.ds, tol.dv

    s_lo, s_hi = max(0, s0 - ds), min(255, s0 + ds)
    v_lo, v_hi = max(0, v0 - dv), min(255, v0 + dv)

    h_lo = (h0 - dh) % 180
    h_hi = (h0 + dh) % 180

    if h_lo <= h_hi:
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        return cv2.inRange(hsv_img, lower, upper)

    # wrap case: [0..h_hi] U [h_lo..179]
    lower1 = np.array([0, s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    lower2 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper2 = np.array([179, s_hi, v_hi], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1), cv2.inRange(hsv_img, lower2, upper2))


def color_ratio_in_box(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    color: HsvColor,
    tol: HsvTol,
) -> float:
    x1, y1, x2, y2 = box_xyxy
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _mask_for_color(hsv, color, tol)
    return float(np.count_nonzero(mask)) / float(mask.size)


def classify_friend_enemy(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    friend: Optional[HsvColor],
    enemy: Optional[HsvColor],
    friend_tol: HsvTol,
    enemy_tol: HsvTol,
    min_ratio: float,
) -> Tuple[str, float]:
    """
    Returns (label, score) where label in {"friend","enemy","unknown"}.
    score is the winning ratio.
    """
    fr = color_ratio_in_box(frame_bgr, box_xyxy, friend, friend_tol) if friend is not None else 0.0
    er = color_ratio_in_box(frame_bgr, box_xyxy, enemy, enemy_tol) if enemy is not None else 0.0

    if fr >= min_ratio and fr >= er:
        return "friend", fr
    if er >= min_ratio and er > fr:
        return "enemy", er
    return "unknown", max(fr, er)


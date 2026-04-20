"""Takip kutusu (ROI) içinde balon/hedef renk için HSV nişangâh noktası (Çelik Kubbe senaryosu)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# Varsayılan morfoloji: gürültüyü kes, küçük delikleri kapat (yuvarlak balon)
_KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def color_aim_in_bbox(
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    hsv_ranges: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    min_area: int,
    *,
    balloon_mode: bool = True,
) -> Optional[Tuple[float, float, float]]:
    """
    ROI içinde bir veya birden fazla HSV aralığının birleşimi ile maske oluşturur.

    balloon_mode=True (önerilen): açılım + kapanım, en büyük dış konturun ağırlık merkezi.
    balloon_mode=False: klasik tam maske moments (tek/çoklu aralık OR ile birleştirilir).

    Dönüş: (cx, cy, kalite) — kalite = kontur_alanı / ROI_alanı ∈ (0,1], balonsa anlamlı.
    """
    if not hsv_ranges:
        return None
    h, w = frame_bgr.shape[:2]
    xi1 = max(0, min(w - 1, int(round(x1))))
    yi1 = max(0, min(h - 1, int(round(y1))))
    xi2 = max(0, min(w, int(round(x2))))
    yi2 = max(0, min(h, int(round(y2))))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    roi = frame_bgr[yi1:yi2, xi1:xi2]
    roi_w = xi2 - xi1
    roi_h = yi2 - yi1
    roi_area = float(max(1, roi_w * roi_h))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for lo, hi in hsv_ranges:
        lower = np.array(lo, dtype=np.uint8)
        upper = np.array(hi, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if balloon_mode:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_CLOSE)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        a = float(cv2.contourArea(best))
        if a < float(min_area):
            return None
        m = cv2.moments(best)
        if m["m00"] <= 1e-6:
            return None
        cx = m["m10"] / m["m00"] + float(xi1)
        cy = m["m01"] / m["m00"] + float(yi1)
        quality = min(1.0, a / roi_area)
        return (cx, cy, quality)

    # Legacy: tüm maske üzerinden moment
    if int(cv2.countNonZero(mask)) < int(min_area):
        return None
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 1e-6:
        return None
    cx = m["m10"] / m["m00"] + float(xi1)
    cy = m["m01"] / m["m00"] + float(yi1)
    quality = min(1.0, float(cv2.countNonZero(mask)) / roi_area)
    return (cx, cy, quality)


def color_centroid_in_bbox(
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    min_pixels: int,
) -> Optional[Tuple[float, float]]:
    """Geriye uyumluluk: tek aralık, balloon_mode kapalı."""
    r = color_aim_in_bbox(
        frame_bgr,
        x1,
        y1,
        x2,
        y2,
        [(hsv_lower, hsv_upper)],
        min_pixels,
        balloon_mode=False,
    )
    if r is None:
        return None
    return (r[0], r[1])

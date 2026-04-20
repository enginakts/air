"""
Düşük / değişken ışık için hızlı ön işleme (OpenCV).

Yaygın yaklaşım: LAB uzayında L kanalına CLAHE (kontrast sınırlı yerel histogram
eşitleme) — renkleri aşırı bozmadan yerel kontrast artırır. Çok karanlıkta isteğe
bağlı hafif Gaussian + gama düzeltmesi.

Bu modül sadece cv2/numpy kullanır; tespit/takip kodundan bağımsızdır.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

IlluminationMode = Literal["off", "auto", "low", "bright"]


def mean_brightness_v(bgr: np.ndarray) -> float:
    """0..255 arası ortalama parlaklık (HSV V kanalı)."""
    if bgr is None or bgr.size == 0:
        return 128.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _gamma_lut(gamma: float) -> np.ndarray:
    g = float(gamma)
    if g <= 1e-6:
        g = 1.0
    inv = 1.0 / g
    return (np.power(np.linspace(0.0, 1.0, 256), inv) * 255.0).astype(np.uint8)


def apply_gamma_bgr(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """gamma>1 genelde görüntüyü aydınlatır (orta tonlar)."""
    if abs(gamma - 1.0) < 1e-3:
        return bgr
    lut = _gamma_lut(gamma)
    b, g, r = cv2.split(bgr)
    b = cv2.LUT(b, lut)
    g = cv2.LUT(g, lut)
    r = cv2.LUT(r, lut)
    return cv2.merge([b, g, r])


def apply_clahe_lab(bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, ch_b = cv2.split(lab)
    t = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(t, t))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, ch_b]), cv2.COLOR_LAB2BGR)


def enhance_bgr(
    bgr: np.ndarray,
    *,
    mode: IlluminationMode,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    denoise_light: bool = True,
) -> np.ndarray:
    """
    `bgr`: BGR uint8 (genelde yatay çevrilmiş kare).
    `mode`:
      - off: dokunma
      - auto: V ortalamasına göre CLAHE + hafif gama
      - low: loş ortamı zorla güçlendir
      - bright: aşırı parlak / düşük kontrast için hafif bastırma + CLAHE
    """
    if mode == "off":
        return bgr

    x = bgr
    v_mean = mean_brightness_v(x)

    clip = float(max(0.5, min(8.0, clahe_clip)))
    tile = int(max(2, min(32, clahe_tile)))

    if mode == "low":
        if denoise_light:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        x = apply_clahe_lab(x, clip=max(clip, 2.5), tile=tile)
        x = apply_gamma_bgr(x, 1.22)
        return x

    if mode == "bright":
        x = apply_clahe_lab(x, clip=max(1.2, min(clip, 2.0)), tile=tile)
        x = apply_gamma_bgr(x, 0.88)
        return x

    # auto
    if v_mean < 45.0:
        if denoise_light:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        x = apply_clahe_lab(x, clip=max(clip, 3.0), tile=tile)
        x = apply_gamma_bgr(x, 1.28)
    elif v_mean < 85.0:
        if denoise_light:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        x = apply_clahe_lab(x, clip=max(clip, 2.2), tile=tile)
        x = apply_gamma_bgr(x, 1.12)
    elif v_mean > 215.0:
        x = apply_clahe_lab(x, clip=min(clip, 1.8), tile=tile)
        x = apply_gamma_bgr(x, 0.92)
    else:
        x = apply_clahe_lab(x, clip=max(1.5, min(clip, 2.2)), tile=tile)

    return x


def flip_and_enhance(
    raw_bgr: np.ndarray,
    *,
    mode: IlluminationMode,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    denoise_light: bool = True,
) -> np.ndarray:
    """Kameradan ham kare → Y flip → ışık düzeltmesi (tespit/takip girdisi)."""
    flipped = cv2.flip(raw_bgr, 1)
    return enhance_bgr(
        flipped,
        mode=mode,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        denoise_light=denoise_light,
    )

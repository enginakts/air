from __future__ import annotations

from dataclasses import dataclass


TARGET_CLASS_NAMES = [
    "balistik_fuze",
    "helikopter",
    "savas_ucagi",
    "mini_micro_iha",
]


@dataclass(frozen=True)
class AppConfig:
    # Detector
    conf: float = 0.35
    iou: float = 0.45
    img_size: int = 960
    device: str | None = None  # "cpu", "cuda:0", etc.

    # DeepSORT
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7

    # UI
    window_name: str = "Celikkube | YOLO + DeepSORT"


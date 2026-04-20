"""
Dost / düşman renk izleme: HSV aralıklarında karedeki piksel oranları (OpenCV H: 0–179).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets


@dataclass
class FriendFoeHSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass
class FriendFoeConfig:
    friend: FriendFoeHSVRange
    enemy: FriendFoeHSVRange


def default_friend_foe_config() -> FriendFoeConfig:
    """Ön ayar: dost ≈ mavi, düşman ≈ turuncu (tipik iç mekân HSV)."""
    return FriendFoeConfig(
        friend=FriendFoeHSVRange(100, 128, 60, 255, 40, 255),
        enemy=FriendFoeHSVRange(4, 22, 80, 255, 100, 255),
    )


def _mask_range(hsv: np.ndarray, lo: FriendFoeHSVRange) -> np.ndarray:
    lower = np.array([lo.h_min, lo.s_min, lo.v_min], dtype=np.uint8)
    upper = np.array([lo.h_max, lo.s_max, lo.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def analyze_friend_foe(bgr: np.ndarray, cfg: FriendFoeConfig, *, max_side: int = 320) -> Tuple[float, float, str]:
    """
    Dönüş: (dost_piksel_%, düşman_piksel_%, baskın_etiket)
    baskın_etiket: 'dost' | 'düşman' | 'belirsiz' | 'veri yok'
    """
    if bgr is None or bgr.size == 0:
        return 0.0, 0.0, "veri yok"
    h0, w0 = bgr.shape[:2]
    m = max(h0, w0)
    if m > max_side:
        scale = max_side / float(m)
        small = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mf = _mask_range(hsv, cfg.friend)
    me = _mask_range(hsv, cfg.enemy)
    total = float(mf.size) if mf.size else 1.0
    pf = 100.0 * float(np.count_nonzero(mf)) / total
    pe = 100.0 * float(np.count_nonzero(me)) / total
    if pf < 0.05 and pe < 0.05:
        return pf, pe, "belirsiz"
    if pf - pe >= 1.5:
        dom = "dost"
    elif pe - pf >= 1.5:
        dom = "düşman"
    else:
        dom = "belirsiz"
    return pf, pe, dom


def _spin_h(parent: QtWidgets.QWidget, val: int) -> QtWidgets.QSpinBox:
    s = QtWidgets.QSpinBox(parent)
    s.setRange(0, 179)
    s.setValue(int(val))
    return s


def _spin_sv(parent: QtWidgets.QWidget, val: int) -> QtWidgets.QSpinBox:
    s = QtWidgets.QSpinBox(parent)
    s.setRange(0, 255)
    s.setValue(int(val))
    return s


class FriendFoeDialog(QtWidgets.QDialog):
    """HSV aralıkları + log periyodu (ms)."""

    def __init__(self, cfg: FriendFoeConfig, interval_ms: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dost / düşman — HSV renk aralığı")
        self.setModal(True)
        self.resize(420, 420)

        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Kamera görüntüsü (işlenmiş kare) HSV uzayında taranır. "
            "Dost ve düşman için ayrı aralıklar; log aralığı ms cinsinden."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        def row_range(title: str, r: FriendFoeHSVRange) -> Tuple[QtWidgets.QSpinBox, ...]:
            g = QtWidgets.QGroupBox(title, self)
            gl = QtWidgets.QGridLayout(g)
            hh0, hh1 = _spin_h(self, r.h_min), _spin_h(self, r.h_max)
            ss0, ss1 = _spin_sv(self, r.s_min), _spin_sv(self, r.s_max)
            vv0, vv1 = _spin_sv(self, r.v_min), _spin_sv(self, r.v_max)
            gl.addWidget(QtWidgets.QLabel("H min"), 0, 0)
            gl.addWidget(hh0, 0, 1)
            gl.addWidget(QtWidgets.QLabel("H max"), 0, 2)
            gl.addWidget(hh1, 0, 3)
            gl.addWidget(QtWidgets.QLabel("S min"), 1, 0)
            gl.addWidget(ss0, 1, 1)
            gl.addWidget(QtWidgets.QLabel("S max"), 1, 2)
            gl.addWidget(ss1, 1, 3)
            gl.addWidget(QtWidgets.QLabel("V min"), 2, 0)
            gl.addWidget(vv0, 2, 1)
            gl.addWidget(QtWidgets.QLabel("V max"), 2, 2)
            gl.addWidget(vv1, 2, 3)
            root.addWidget(g)
            return hh0, hh1, ss0, ss1, vv0, vv1

        self._f = row_range("Dost (ör. mavi)", cfg.friend)
        self._e = row_range("Düşman (ör. turuncu)", cfg.enemy)

        row_ms = QtWidgets.QHBoxLayout()
        row_ms.addWidget(QtWidgets.QLabel("Log / terminal çıktı aralığı (ms):"))
        self.sp_interval = QtWidgets.QSpinBox()
        self.sp_interval.setRange(50, 10_000)
        self.sp_interval.setSingleStep(50)
        self.sp_interval.setValue(int(interval_ms))
        self.sp_interval.setToolTip("Çok düşük değer logu ve FPS’i yorar; 300–800 ms tipik.")
        row_ms.addWidget(self.sp_interval)
        row_ms.addStretch(1)
        root.addLayout(row_ms)

        preset = QtWidgets.QPushButton("Ön ayar: mavi dost / turuncu düşman")
        preset.clicked.connect(self._apply_preset)
        root.addWidget(preset)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _apply_preset(self) -> None:
        d = default_friend_foe_config()
        self._fill(self._f, d.friend)
        self._fill(self._e, d.enemy)

    def _fill(self, spins: Tuple[QtWidgets.QSpinBox, ...], r: FriendFoeHSVRange) -> None:
        spins[0].setValue(r.h_min)
        spins[1].setValue(r.h_max)
        spins[2].setValue(r.s_min)
        spins[3].setValue(r.s_max)
        spins[4].setValue(r.v_min)
        spins[5].setValue(r.v_max)

    def _read(self, spins: Tuple[QtWidgets.QSpinBox, ...]) -> FriendFoeHSVRange:
        h0, h1 = spins[0].value(), spins[1].value()
        if h0 > h1:
            h0, h1 = h1, h0
        s0, s1 = spins[2].value(), spins[3].value()
        if s0 > s1:
            s0, s1 = s1, s0
        v0, v1 = spins[4].value(), spins[5].value()
        if v0 > v1:
            v0, v1 = v1, v0
        return FriendFoeHSVRange(h0, h1, s0, s1, v0, v1)

    def result_config(self) -> Tuple[FriendFoeConfig, int]:
        return FriendFoeConfig(friend=self._read(self._f), enemy=self._read(self._e)), int(self.sp_interval.value())

import sys
import time
import os
from dataclasses import asdict
from typing import Optional, Tuple

# OpenCV log (env bazı sürümlerde yorumlanır); asıl sessizlik import sonrası utils.logging ile.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import serial
    import serial.tools.list_ports
except Exception:  # pragma: no cover
    serial = None

from friend_foe import FriendFoeDialog, analyze_friend_foe, default_friend_foe_config
from illumination import flip_and_enhance
from tracker_core import (
    TrackerConfig,
    TrackerEngine,
    apply_auto_heuristics,
    autotune_config,
    default_local_model,
)

try:
    import cv2.utils.logging as cvlog

    if sys.platform == "win32":
        cvlog.setLogLevel(cvlog.LOG_LEVEL_SILENT)
    else:
        cvlog.setLogLevel(cvlog.LOG_LEVEL_ERROR)
except Exception:
    pass


def cv_bgr_to_qimage(frame_bgr: np.ndarray) -> QtGui.QImage:
    # QImage, NumPy buffer'ın contiguous ve doğru stride ile olmasını ister.
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    bytes_per_line = int(rgb.strides[0])
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class VideoWidget(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("videoView")
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setText("Kamera kapalı")

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        img = cv_bgr_to_qimage(frame_bgr)
        pix = QtGui.QPixmap.fromImage(img)
        self.setText("")
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self.pixmap() is not None:
            self.setPixmap(
                self.pixmap().scaled(
                    self.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.FastTransformation,
                )
            )
        super().resizeEvent(event)


class HsvRangeDialog(QtWidgets.QDialog):
    """Çelik Kubbe — balon/hedef için HSV (tek veya çift aralık, ikinci çift genelde kırmızı H için)."""

    def __init__(
        self,
        lo: Tuple[int, int, int],
        hi: Tuple[int, int, int],
        min_px: int,
        *,
        lo2: Optional[Tuple[int, int, int]] = None,
        hi2: Optional[Tuple[int, int, int]] = None,
        dual_initial: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("HSV — balon / hedef renk")
        self.setModal(True)
        root = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "OpenCV HSV: H∈[0,179], S,V∈[0,255]. Balon modu en büyük renk kümesinin merkezini nişangâh alır. "
            "Kırmızı için genelde iki HSV aralığı gerekir (hue uçları)."
        )
        tip.setWordWrap(True)
        root.addWidget(tip)

        form = QtWidgets.QFormLayout()
        self.sp_h0 = QtWidgets.QSpinBox()
        self.sp_h0.setRange(0, 179)
        self.sp_s0 = QtWidgets.QSpinBox()
        self.sp_s0.setRange(0, 255)
        self.sp_v0 = QtWidgets.QSpinBox()
        self.sp_v0.setRange(0, 255)
        self.sp_h1 = QtWidgets.QSpinBox()
        self.sp_h1.setRange(0, 179)
        self.sp_s1 = QtWidgets.QSpinBox()
        self.sp_s1.setRange(0, 255)
        self.sp_v1 = QtWidgets.QSpinBox()
        self.sp_v1.setRange(0, 255)
        self.sp_min_px = QtWidgets.QSpinBox()
        self.sp_min_px.setToolTip("En az kontur alanı (balon küçükse düşürün)")
        self.sp_min_px.setRange(10, 50000)
        self.sp_h0.setValue(int(lo[0]))
        self.sp_s0.setValue(int(lo[1]))
        self.sp_v0.setValue(int(lo[2]))
        self.sp_h1.setValue(int(hi[0]))
        self.sp_s1.setValue(int(hi[1]))
        self.sp_v1.setValue(int(hi[2]))
        self.sp_min_px.setValue(int(min_px))
        form.addRow("Aralık 1 — Alt H,S,V", self._row_hsv(self.sp_h0, self.sp_s0, self.sp_v0))
        form.addRow("Aralık 1 — Üst H,S,V", self._row_hsv(self.sp_h1, self.sp_s1, self.sp_v1))
        form.addRow("Min. kontur alanı", self.sp_min_px)

        self.cb_dual = QtWidgets.QCheckBox("İkinci HSV aralığı kullan (ör. kırmızının yüksek H kolu)")
        rd = lo2 if lo2 is not None else (170, 120, 70)
        ru = hi2 if hi2 is not None else (179, 255, 255)
        self.cb_dual.setChecked(bool(dual_initial))
        self.sp2_h0 = QtWidgets.QSpinBox()
        self.sp2_h0.setRange(0, 179)
        self.sp2_s0 = QtWidgets.QSpinBox()
        self.sp2_s0.setRange(0, 255)
        self.sp2_v0 = QtWidgets.QSpinBox()
        self.sp2_v0.setRange(0, 255)
        self.sp2_h1 = QtWidgets.QSpinBox()
        self.sp2_h1.setRange(0, 179)
        self.sp2_s1 = QtWidgets.QSpinBox()
        self.sp2_s1.setRange(0, 255)
        self.sp2_v1 = QtWidgets.QSpinBox()
        self.sp2_v1.setRange(0, 255)
        self.sp2_h0.setValue(int(rd[0]))
        self.sp2_s0.setValue(int(rd[1]))
        self.sp2_v0.setValue(int(rd[2]))
        self.sp2_h1.setValue(int(ru[0]))
        self.sp2_s1.setValue(int(ru[1]))
        self.sp2_v1.setValue(int(ru[2]))

        self.r2_frame = QtWidgets.QWidget()
        r2f = QtWidgets.QFormLayout(self.r2_frame)
        r2f.setContentsMargins(0, 8, 0, 0)
        r2f.addRow("Aralık 2 — Alt H,S,V", self._row_hsv(self.sp2_h0, self.sp2_s0, self.sp2_v0))
        r2f.addRow("Aralık 2 — Üst H,S,V", self._row_hsv(self.sp2_h1, self.sp2_s1, self.sp2_v1))

        root.addLayout(form)
        root.addWidget(self.cb_dual)
        root.addWidget(self.r2_frame)

        self.cb_dual.toggled.connect(self._toggle_r2)

        preset = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Tek: turuncu-kırmızı low-H")
        b1.clicked.connect(self._preset_red_single)
        b3 = QtWidgets.QPushButton("Çift H: tam kırmızı")
        b3.clicked.connect(self._preset_red_dual)
        b2 = QtWidgets.QPushButton("Tek: sarı-yeşil")
        b2.clicked.connect(self._preset_yellow_green)
        preset.addWidget(b1)
        preset.addWidget(b3)
        preset.addWidget(b2)
        root.addLayout(preset)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        ok = QtWidgets.QPushButton("Tamam")
        ok.clicked.connect(self.accept)
        ca = QtWidgets.QPushButton("İptal")
        ca.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(ca)
        root.addLayout(row)
        self.resize(460, 520)
        self._toggle_r2(bool(dual_initial))

    def _toggle_r2(self, on: bool) -> None:
        self.r2_frame.setVisible(on)

    @staticmethod
    def _row_hsv(h: QtWidgets.QSpinBox, s: QtWidgets.QSpinBox, v: QtWidgets.QSpinBox) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(h)
        lay.addWidget(s)
        lay.addWidget(v)
        return w

    def _preset_red_single(self) -> None:
        self.cb_dual.setChecked(False)
        self.sp_h0.setValue(0)
        self.sp_s0.setValue(120)
        self.sp_v0.setValue(70)
        self.sp_h1.setValue(15)
        self.sp_s1.setValue(255)
        self.sp_v1.setValue(255)

    def _preset_red_dual(self) -> None:
        self.cb_dual.setChecked(True)
        self.sp_h0.setValue(0)
        self.sp_s0.setValue(120)
        self.sp_v0.setValue(70)
        self.sp_h1.setValue(10)
        self.sp_s1.setValue(255)
        self.sp_v1.setValue(255)
        self.sp2_h0.setValue(170)
        self.sp2_s0.setValue(120)
        self.sp2_v0.setValue(70)
        self.sp2_h1.setValue(179)
        self.sp2_s1.setValue(255)
        self.sp2_v1.setValue(255)

    def _preset_yellow_green(self) -> None:
        self.cb_dual.setChecked(False)
        self.sp_h0.setValue(22)
        self.sp_s0.setValue(60)
        self.sp_v0.setValue(60)
        self.sp_h1.setValue(95)
        self.sp_s1.setValue(255)
        self.sp_v1.setValue(255)

    def result_full(
        self,
    ) -> Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        int,
        Optional[Tuple[int, int, int]],
        Optional[Tuple[int, int, int]],
        bool,
    ]:
        lo = (int(self.sp_h0.value()), int(self.sp_s0.value()), int(self.sp_v0.value()))
        hi = (int(self.sp_h1.value()), int(self.sp_s1.value()), int(self.sp_v1.value()))
        dual = self.cb_dual.isChecked()
        if dual:
            lo2 = (int(self.sp2_h0.value()), int(self.sp2_s0.value()), int(self.sp2_v0.value()))
            hi2 = (int(self.sp2_h1.value()), int(self.sp2_s1.value()), int(self.sp2_v1.value()))
            return lo, hi, int(self.sp_min_px.value()), lo2, hi2, True
        return lo, hi, int(self.sp_min_px.value()), None, None, False


class _EngineInitSignals(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)  # TrackerConfig
    failed = QtCore.pyqtSignal(str)


class _EngineInitWorker(QtCore.QRunnable):
    def __init__(self, cfg: TrackerConfig, cap: cv2.VideoCapture, do_autotune: bool, signals: _EngineInitSignals) -> None:
        super().__init__()
        self.cfg = cfg
        self.cap = cap
        self.do_autotune = do_autotune
        self.signals = signals

    def run(self) -> None:
        try:
            self.signals.status.emit("[MODEL] yükleniyor...")
            tmp = TrackerEngine(self.cfg)
            cfg2 = self.cfg
            if self.do_autotune:
                self.signals.status.emit("[AUTOTUNE] çalışıyor...")
                cfg2 = autotune_config(cfg2, tmp.model, self.cap, tune_seconds=4.0, target_fps=18.0)
            self.signals.status.emit("[MODEL] hazır")
            self.signals.done.emit(cfg2)
        except Exception as e:
            self.signals.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Çelik Kubbe · balon nişangâhı")
        self.setMinimumSize(1200, 700)

        self.cap: Optional[cv2.VideoCapture] = None
        self.engine: Optional[TrackerEngine] = None
        self._engine_thread_pool = QtCore.QThreadPool.globalInstance()
        self._engine_signals: Optional[_EngineInitSignals] = None
        self._engine_pending_cfg: Optional[TrackerConfig] = None
        self.last_frame_time = time.time()
        self.fps_smooth = 0.0

        self._ff_cfg = default_friend_foe_config()
        self._ff_interval_ms = 400
        self._ff_last_log_ts = 0.0

        self._hsv_lo: Tuple[int, int, int] = (0, 100, 100)
        self._hsv_hi: Tuple[int, int, int] = (10, 255, 255)
        self._hsv_lo2: Optional[Tuple[int, int, int]] = None
        self._hsv_hi2: Optional[Tuple[int, int, int]] = None
        self._hsv_dual: bool = False
        self._color_min_px: int = 80
        self._settings_dialog: Optional[QtWidgets.QDialog] = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_ports()
        self._refresh_cameras()
        QtCore.QTimer.singleShot(0, self._maybe_autostart_camera)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        # Left panel (kontroller + log — log alanı dikeyde daha geniş pay alır)
        left = QtWidgets.QFrame()
        left.setObjectName("leftPanel")
        left.setMinimumWidth(380)
        left.setMaximumWidth(520)
        left_l = QtWidgets.QVBoxLayout(left)
        left_l.setContentsMargins(10, 10, 10, 10)

        logo = QtWidgets.QLabel("Yaman")
        logo.setObjectName("logo")
        left_l.addWidget(logo)

        row_settings_btn = QtWidgets.QHBoxLayout()
        self.btn_tracking_settings = QtWidgets.QPushButton("Takip ve görüntü ayarları…")
        self.btn_tracking_settings.setObjectName("secondaryOutlineBtn")
        self.btn_tracking_settings.setToolTip("Model, tracker, ışık ve çıkarım parametreleri (ayrı pencere)")
        self.btn_tracking_settings.clicked.connect(self._open_tracking_settings)
        row_settings_btn.addWidget(self.btn_tracking_settings)
        left_l.addLayout(row_settings_btn)

        # Kontrol (sol panel — kompakt)
        tab_control = QtWidgets.QWidget()
        ctl = QtWidgets.QVBoxLayout(tab_control)
        ctl.setContentsMargins(0, 0, 0, 0)

        self.btn_arm = QtWidgets.QPushButton("ARM")
        self.btn_arm.setObjectName("primaryBtn")
        self.btn_arm.setFixedHeight(40)
        self.btn_arm.clicked.connect(lambda: self._log("[ARM] toggle"))
        ctl.addWidget(self.btn_arm)

        ctl.addSpacing(8)
        self.btn_start_cam = QtWidgets.QPushButton("Kamerayı Başlat")
        self.btn_start_cam.setObjectName("primaryBtn")
        self.btn_start_cam.clicked.connect(self.toggle_camera)
        ctl.addWidget(self.btn_start_cam)

        row_ff = QtWidgets.QHBoxLayout()
        self.btn_ff = QtWidgets.QPushButton("Dost/Düşman izle")
        self.btn_ff.setCheckable(True)
        self.btn_ff.setToolTip("Açıkken HSV renk aralığına göre kare analizi; log + terminal periyodik çıktı")
        self.btn_ff.toggled.connect(self._on_friend_foe_toggled)
        self.btn_ff_settings = QtWidgets.QPushButton("Ayar…")
        self.btn_ff_settings.setToolTip("Dost/düşman HSV aralığı ve log süresi (izleme açıkken de uygulanır)")
        self.btn_ff_settings.clicked.connect(self._open_friend_foe_settings)
        row_ff.addWidget(self.btn_ff, 1)
        row_ff.addWidget(self.btn_ff_settings)
        ctl.addLayout(row_ff)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        btns = {}
        for name in ["↖", "↑", "↗", "←", "•", "→", "↙", "↓", "↘"]:
            b = QtWidgets.QPushButton(name)
            b.setFixedSize(52, 52)
            b.clicked.connect(lambda _, t=name: self._log(f"[MOVE] {t}"))
            btns[name] = b
        grid.addWidget(btns["↖"], 0, 0)
        grid.addWidget(btns["↑"], 0, 1)
        grid.addWidget(btns["↗"], 0, 2)
        grid.addWidget(btns["←"], 1, 0)
        grid.addWidget(btns["•"], 1, 1)
        grid.addWidget(btns["→"], 1, 2)
        grid.addWidget(btns["↙"], 2, 0)
        grid.addWidget(btns["↓"], 2, 1)
        grid.addWidget(btns["↘"], 2, 2)
        ctl.addLayout(grid)
        ctl.addStretch(1)

        left_l.addWidget(tab_control, 2)

        row_color = QtWidgets.QHBoxLayout()
        self.chk_box_color = QtWidgets.QCheckBox("Balon / hedef renk nişangâhı")
        self.chk_box_color.setToolTip(
            "Çelik Kubbe: takip kutusu içinde HSV ile balon boyası aranır; "
            "bulunduğu en büyük renk bölgesinin merkezi cyan artı ile gösterilir; [AIM] ile px ve norm koordinat loglanır."
        )
        self.btn_hsv_range = QtWidgets.QPushButton("HSV aralığı…")
        self.btn_hsv_range.clicked.connect(self._open_hsv_range_dialog)
        row_color.addWidget(self.chk_box_color, 1)
        row_color.addWidget(self.btn_hsv_range)
        left_l.addLayout(row_color)

        # Ayar formu: ana pencerede gösterilmez; diyalog içinde QScrollArea ile açılır
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setObjectName("trackingSettingsPanel")
        self.settings_panel.setParent(self)
        self.settings_panel.hide()
        stl = QtWidgets.QFormLayout(self.settings_panel)
        stl.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        stl.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.cb_backend = QtWidgets.QComboBox()
        self.cb_backend.addItem("DeepSORT (REID+Kalman)", "deepsort")
        self.cb_backend.addItem("ByteTrack (Kalman)", "bytetrack")
        self.cb_backend.addItem("BoT-SORT (önerilen, ID daha stabil)", "botsort")
        self.cb_backend.setCurrentIndex(2)
        stl.addRow("tracker", self.cb_backend)

        self.sp_max_age = QtWidgets.QSpinBox()
        self.sp_max_age.setRange(15, 120)
        self.sp_max_age.setValue(50)
        self.sp_max_age.setToolTip("DeepSORT: kayıp kare sonrası izi tutma süresi (↑ = kopmada aynı ID şansı)")
        stl.addRow("DS max_age", self.sp_max_age)

        self.sp_n_init = QtWidgets.QSpinBox()
        self.sp_n_init.setRange(2, 12)
        self.sp_n_init.setValue(4)
        self.sp_n_init.setToolTip("DeepSORT: onaylı iz öncesi gerekli ardışık eşleşme kare sayısı")
        stl.addRow("DS n_init", self.sp_n_init)

        self.cb_ds_embed = QtWidgets.QComboBox()
        self.cb_ds_embed.addItem("MobileNet (hızlı, CPU/GPU)", "mobilenet")
        self.cb_ds_embed.addItem("CLIP ViT-B/32 (GPU, daha iyi görünüm)", "clip_ViT-B/32")
        stl.addRow("DS gömme", self.cb_ds_embed)

        self.sp_ds_iou_gate = QtWidgets.QDoubleSpinBox()
        self.sp_ds_iou_gate.setRange(0.5, 0.98)
        self.sp_ds_iou_gate.setSingleStep(0.02)
        self.sp_ds_iou_gate.setValue(0.88)
        self.sp_ds_iou_gate.setToolTip("DeepSORT: hızlı hareket için IoU kapısı (yüksek = daha toleranslı)")
        stl.addRow("DS max_iou_distance", self.sp_ds_iou_gate)

        self.sp_ds_cos = QtWidgets.QDoubleSpinBox()
        self.sp_ds_cos.setRange(0.15, 0.55)
        self.sp_ds_cos.setSingleStep(0.01)
        self.sp_ds_cos.setValue(0.38)
        self.sp_ds_cos.setToolTip("DeepSORT: görünüm benzerliği (↑ = hafif bulanıkta aynı ID)")
        stl.addRow("DS max_cosine_distance", self.sp_ds_cos)

        self.sp_ds_nms = QtWidgets.QDoubleSpinBox()
        self.sp_ds_nms.setRange(0.35, 1.0)
        self.sp_ds_nms.setSingleStep(0.05)
        self.sp_ds_nms.setValue(0.55)
        self.sp_ds_nms.setToolTip("DeepSORT: 1.0=NMS kapalı; düşük=çift kutuları azaltır")
        stl.addRow("DS nms_max_overlap", self.sp_ds_nms)

        self.cb_ds_gating_xy = QtWidgets.QCheckBox("DeepSORT: sadece konum gating (hızlı el hareketi)")
        self.cb_ds_gating_xy.setChecked(True)
        stl.addRow("", self.cb_ds_gating_xy)

        self.cb_autotune = QtWidgets.QCheckBox("Autotune (başlangıç, ~4 sn)")
        self.cb_autotune.setChecked(False)
        stl.addRow("", self.cb_autotune)

        self.cb_illumination = QtWidgets.QComboBox()
        self.cb_illumination.addItem("Otomatik (önerilen)", "auto")
        self.cb_illumination.addItem("Kapalı", "off")
        self.cb_illumination.addItem("Loş / az ışık", "low")
        self.cb_illumination.addItem("Parlak / aşırı exposure", "bright")
        stl.addRow("Işık", self.cb_illumination)

        self.sp_clahe_clip = QtWidgets.QDoubleSpinBox()
        self.sp_clahe_clip.setRange(0.5, 8.0)
        self.sp_clahe_clip.setSingleStep(0.1)
        self.sp_clahe_clip.setValue(2.0)
        stl.addRow("clahe_clip", self.sp_clahe_clip)

        self.sp_clahe_tile = QtWidgets.QSpinBox()
        self.sp_clahe_tile.setRange(2, 32)
        self.sp_clahe_tile.setValue(8)
        stl.addRow("clahe_tile", self.sp_clahe_tile)

        self.cb_illum_denoise = QtWidgets.QCheckBox("Loş ışıkta hafif gürültü azaltma (Gaussian)")
        self.cb_illum_denoise.setChecked(True)
        stl.addRow("", self.cb_illum_denoise)

        self.sp_imgsz = QtWidgets.QSpinBox()
        self.sp_imgsz.setRange(320, 1920)
        self.sp_imgsz.setSingleStep(32)
        self.sp_imgsz.setValue(640)
        stl.addRow("imgsz", self.sp_imgsz)

        self.sp_tile = QtWidgets.QSpinBox()
        self.sp_tile.setRange(1, 3)
        self.sp_tile.setValue(1)
        stl.addRow("tile", self.sp_tile)

        self.sp_infer_every = QtWidgets.QSpinBox()
        self.sp_infer_every.setRange(1, 10)
        self.sp_infer_every.setValue(1)
        self.sp_infer_every.setToolTip("DeepSORT: YOLO her N karede bir (1=her karede tespit çizimi; 2+=FPS↑)")
        stl.addRow("infer_every", self.sp_infer_every)

        self.sp_max_det = QtWidgets.QSpinBox()
        self.sp_max_det.setRange(10, 300)
        self.sp_max_det.setValue(80)
        self.sp_max_det.setToolTip("YOLO max_det: düşük değer NMS sonrası yükü azaltır (FPS↑)")
        stl.addRow("max_det", self.sp_max_det)

        self.sp_conf = QtWidgets.QDoubleSpinBox()
        self.sp_conf.setRange(0.01, 0.99)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.setValue(0.25)
        stl.addRow("conf", self.sp_conf)

        self.sp_iou = QtWidgets.QDoubleSpinBox()
        self.sp_iou.setRange(0.05, 0.95)
        self.sp_iou.setSingleStep(0.05)
        self.sp_iou.setValue(0.5)
        stl.addRow("iou", self.sp_iou)

        self.sp_min_area = QtWidgets.QSpinBox()
        self.sp_min_area.setRange(0, 20000)
        self.sp_min_area.setValue(0)
        stl.addRow("min_box_area", self.sp_min_area)

        # Log panel
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(300)
        left_l.addWidget(self.log, 6)

        # Right panel (top selectors + video)
        right_container = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)

        top = QtWidgets.QFrame()
        top.setObjectName("topBar")
        top_l = QtWidgets.QHBoxLayout(top)
        top_l.setContentsMargins(10, 6, 10, 6)

        self.cb_baud = QtWidgets.QComboBox()
        for b in [9600, 19200, 38400, 57600, 115200, 230400]:
            self.cb_baud.addItem(str(b))
        self.cb_baud.setCurrentText("9600")

        self.cb_port = QtWidgets.QComboBox()
        self.btn_refresh_ports = QtWidgets.QToolButton()
        self.btn_refresh_ports.setText("⟳")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)

        self.cb_camera = QtWidgets.QComboBox()
        self.btn_refresh_cam = QtWidgets.QToolButton()
        self.btn_refresh_cam.setText("📷")
        self.btn_refresh_cam.clicked.connect(self._refresh_cameras)

        self.btn_link = QtWidgets.QToolButton()
        self.btn_link.setText("🔗")
        self.btn_link.clicked.connect(self._connect_serial)

        top_l.addWidget(QtWidgets.QLabel("BaudRate:"))
        top_l.addWidget(self.cb_baud)
        top_l.addSpacing(10)
        top_l.addWidget(QtWidgets.QLabel("COM Port:"))
        top_l.addWidget(self.cb_port, 1)
        top_l.addWidget(self.btn_refresh_ports)
        top_l.addSpacing(10)
        top_l.addWidget(QtWidgets.QLabel("Kamera:"))
        top_l.addWidget(self.cb_camera, 1)
        top_l.addWidget(self.btn_refresh_cam)
        top_l.addSpacing(10)
        top_l.addWidget(self.btn_link)

        right.addWidget(top)

        self.video = VideoWidget()
        right.addWidget(self.video, 1)

        footer = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Hazır")
        self.lbl_status.setStyleSheet("color:#444;")
        self.lbl_fps = QtWidgets.QLabel("FPS: -")
        self.lbl_fps.setStyleSheet("color:#444;")
        footer.addWidget(self.lbl_status, 1)
        footer.addWidget(self.lbl_fps)
        right.addLayout(footer)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([430, 980])
        right_container.setMinimumWidth(700)

        root.addWidget(splitter, 1)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f3f3f3; }

            /* Left panel */
            QFrame#leftPanel { background:#161616; border:1px solid #2a2a2a; }
            QLabel#logo { font-size:28px; font-weight:900; color:#f7d23e; letter-spacing:1px; }
            QPushButton#secondaryOutlineBtn {
                background:#252525;
                color:#f7d23e;
                border:1px solid #4a4a4a;
                padding:8px 10px;
                font-weight:700;
            }
            QPushButton#secondaryOutlineBtn:hover { background:#333; border-color:#666; }

            QRadioButton, QCheckBox { color:#e6e6e6; font-size:12px; }
            QRadioButton::indicator, QCheckBox::indicator { width:14px; height:14px; }

            QPushButton#primaryBtn { background:#f7d23e; color:#111; border:1px solid #caa92c; padding:10px; font-weight:900; }
            QPushButton#primaryBtn:hover { background:#ffe27a; }
            QPushButton { background:#f7d23e; color:#111; border:1px solid #caa92c; padding:8px; font-weight:800; }
            QPushButton:hover { background:#ffe27a; }

            QPlainTextEdit#logView { background:#0e0e0e; color:#d8d8d8; border:1px solid #2a2a2a; font-family:Consolas; font-size:12px; padding:6px; }

            /* Takip ayarları diyaloğu (QFormLayout) */
            QWidget#trackingSettingsPanel { background:#161616; }
            QWidget#trackingSettingsPanel QLabel { color:#f0f0f0; font-weight:800; }
            QWidget#trackingSettingsPanel QComboBox,
            QWidget#trackingSettingsPanel QSpinBox,
            QWidget#trackingSettingsPanel QDoubleSpinBox {
                color:#111;
                background:#f6f6f6;
                border:1px solid #cfcfcf;
                padding:4px 8px;
                min-height:28px;
            }
            QWidget#trackingSettingsPanel QCheckBox {
                color:#f0f0f0;
                background: transparent;
                border:0;
                padding:2px 0;
                min-height:22px;
            }
            QWidget#trackingSettingsPanel QCheckBox::indicator {
                width:16px;
                height:16px;
                border:1px solid #cfcfcf;
                background:#f6f6f6;
            }
            QWidget#trackingSettingsPanel QCheckBox::indicator:checked {
                background:#f7d23e;
                border:1px solid #caa92c;
            }
            QWidget#trackingSettingsPanel QComboBox::drop-down { border:0; width:22px; }
            QWidget#trackingSettingsPanel QComboBox QAbstractItemView {
                background:#ffffff;
                color:#111;
                selection-background-color:#f7d23e;
                selection-color:#111;
            }

            QDialog#trackingSettingsDialog { background:#1a1a1a; }
            QDialog#trackingSettingsDialog QScrollArea { border:1px solid #333; background:#161616; }

            /* Right side top bar */
            QFrame#topBar { background:#efefef; border:1px solid #d0d0d0; }
            QFrame#topBar QLabel { color:#222; font-weight:700; }
            QFrame#topBar QComboBox { background:#ffffff; color:#111; padding:4px 8px; border:1px solid #cfcfcf; min-height:26px; }
            QFrame#topBar QToolButton { background:#f7d23e; color:#111; border:1px solid #caa92c; padding:6px 10px; font-weight:900; min-height:26px; }
            QFrame#topBar QToolButton:hover { background:#ffe27a; }

            /* Video view */
            QLabel#videoView { background:#0b0b0b; color:#bdbdbd; border:1px solid #333; font-size:16px; }

            /* Inputs (sol panelde Ayarlar dışında kalanlar için genel) */
            QSpinBox, QDoubleSpinBox, QComboBox { background:#1f1f1f; color:#eee; border:1px solid #333; padding:4px 8px; min-height:26px; }
            """
        )

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def _open_tracking_settings(self) -> None:
        d = self._ensure_settings_dialog()
        d.exec_()

    def _ensure_settings_dialog(self) -> QtWidgets.QDialog:
        if self._settings_dialog is not None:
            return self._settings_dialog
        d = QtWidgets.QDialog(self)
        d.setObjectName("trackingSettingsDialog")
        d.setWindowTitle("Yaman — Takip ve görüntü ayarları")
        d.setModal(True)
        lay = QtWidgets.QVBoxLayout(d)
        lay.setContentsMargins(12, 12, 12, 12)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.settings_panel)
        self.settings_panel.show()
        lay.addWidget(scroll, 1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        hb = QtWidgets.QPushButton("Kapat")
        hb.setDefault(True)
        hb.clicked.connect(d.accept)
        row.addWidget(hb)
        lay.addLayout(row)
        d.resize(520, 720)
        self._settings_dialog = d
        return d

    def _open_hsv_range_dialog(self) -> None:
        dlg = HsvRangeDialog(
            self._hsv_lo,
            self._hsv_hi,
            self._color_min_px,
            lo2=self._hsv_lo2,
            hi2=self._hsv_hi2,
            dual_initial=self._hsv_dual,
            parent=self,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            lo, hi, px, lo2, hi2, dual = dlg.result_full()
            self._hsv_lo, self._hsv_hi, self._color_min_px = lo, hi, px
            self._hsv_dual = dual
            self._hsv_lo2, self._hsv_hi2 = (lo2, hi2) if dual else (None, None)
            extra = f" aralık2={self._hsv_lo2}/{self._hsv_hi2}" if dual else ""
            self._log(f"[AIM] HSV güncellendi r1={self._hsv_lo}-{self._hsv_hi} min_alan={self._color_min_px}{extra}")

    def _sync_engine_color_from_ui(self) -> None:
        if self.engine is None:
            return
        u = self._current_cfg()
        e = self.engine.cfg
        e.box_color_focus_enabled = u.box_color_focus_enabled
        e.box_hsv_lower = u.box_hsv_lower
        e.box_hsv_upper = u.box_hsv_upper
        e.box_hsv_lower2 = u.box_hsv_lower2
        e.box_hsv_upper2 = u.box_hsv_upper2
        e.box_color_min_pixels = u.box_color_min_pixels
        e.box_color_balloon_mode = u.box_color_balloon_mode
        e.box_color_log_interval_sec = u.box_color_log_interval_sec

    def _refresh_ports(self) -> None:
        self.cb_port.clear()
        ports = []
        if serial is not None:
            try:
                ports = list(serial.tools.list_ports.comports())
            except Exception:
                ports = []
        if not ports:
            self.cb_port.addItem("COM1")
            self.cb_port.addItem("COM2")
        else:
            for p in ports:
                self.cb_port.addItem(p.device)
        self._log("[UI] COM port listesi yenilendi")

    def _camera_index_available(self, i: int) -> bool:
        """Windows’ta `VideoCapture(i)` varsayılan arka uç Orbbec obsensor hatası basabiliyor; DSHOW/MSMF ile dene."""
        if sys.platform == "win32":
            for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                cap = cv2.VideoCapture(i, api)
                try:
                    if cap.isOpened():
                        return True
                finally:
                    cap.release()
            return False
        cap = cv2.VideoCapture(i)
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()

    def _refresh_cameras(self) -> None:
        self.cb_camera.clear()
        found = []
        for i in range(5):
            if self._camera_index_available(i):
                found.append(i)
        if not found:
            self.cb_camera.addItem("0")
        else:
            for i in found:
                self.cb_camera.addItem(str(i))
        self._log("[UI] Kamera listesi yenilendi")

    def _connect_serial(self) -> None:
        port = self.cb_port.currentText()
        baud = self.cb_baud.currentText()
        self._log(f"[SERIAL] connect requested: {port} @ {baud}")
        self.lbl_status.setText(f"Serial: {port} @ {baud} (demo)")

    def _open_friend_foe_settings(self) -> None:
        dlg = FriendFoeDialog(self._ff_cfg, self._ff_interval_ms, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._ff_cfg, self._ff_interval_ms = dlg.result_config()
            self._log(f"[FF] HSV aralığı güncellendi; çıktı aralığı {self._ff_interval_ms} ms")

    def _on_friend_foe_toggled(self, checked: bool) -> None:
        if checked:
            dlg = FriendFoeDialog(self._ff_cfg, self._ff_interval_ms, parent=self)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                self.btn_ff.blockSignals(True)
                self.btn_ff.setChecked(False)
                self.btn_ff.blockSignals(False)
                return
            self._ff_cfg, self._ff_interval_ms = dlg.result_config()
            self._ff_last_log_ts = 0.0
            self._log("[FF] izleme açık — log paneli + terminal (stdout) periyodik")
        else:
            self._log("[FF] izleme kapalı")

    def _open_capture(self, cam_idx: int) -> Tuple[Optional[cv2.VideoCapture], str]:
        # Bazı cihazlarda DSHOW açılır ama frame gelmez; MSMF daha iyi olabilir.
        # Windows’ta `VideoCapture(idx)` (ANY) obsensor + "Camera index out of range" spamına yol açabiliyor — kullanma.
        candidates = [
            ("DSHOW", lambda: cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)),
            ("MSMF", lambda: cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)),
        ]
        if sys.platform != "win32":
            candidates.append(("ANY", lambda: cv2.VideoCapture(cam_idx)))
        for name, factory in candidates:
            cap = factory()
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap, name
        return None, "NONE"

    def _warmup_read(self, cap: cv2.VideoCapture, tries: int = 30) -> Tuple[bool, Optional[np.ndarray]]:
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True, frame
            time.sleep(0.02)
        return False, None

    def _current_cfg(self) -> TrackerConfig:
        # Ağırlıklar `object_tracking/` klasöründe aranır (cwd’den bağımsız).
        model = default_local_model()
        return TrackerConfig(
            model=model,
            imgsz=int(self.sp_imgsz.value()),
            conf=float(self.sp_conf.value()),
            iou=float(self.sp_iou.value()),
            tile=int(self.sp_tile.value()),
            min_box_area=int(self.sp_min_area.value()),
            infer_every=max(1, int(self.sp_infer_every.value())),
            max_det=max(1, int(self.sp_max_det.value())),
            max_age=int(self.sp_max_age.value()),
            n_init=int(self.sp_n_init.value()),
            flip_y=False,
            backend=str(self.cb_backend.currentData() or "botsort"),
            deepsort_embedder=str(self.cb_ds_embed.currentData() or "mobilenet"),
            max_iou_distance=float(self.sp_ds_iou_gate.value()),
            max_cosine_distance=float(self.sp_ds_cos.value()),
            nms_max_overlap=float(self.sp_ds_nms.value()),
            gating_only_position=bool(self.cb_ds_gating_xy.isChecked()),
            illumination=str(self.cb_illumination.currentData() or "auto"),
            clahe_clip=float(self.sp_clahe_clip.value()),
            clahe_tile=int(self.sp_clahe_tile.value()),
            illum_denoise=bool(self.cb_illum_denoise.isChecked()),
            box_color_focus_enabled=bool(self.chk_box_color.isChecked()),
            box_hsv_lower=tuple(int(x) for x in self._hsv_lo),
            box_hsv_upper=tuple(int(x) for x in self._hsv_hi),
            box_hsv_lower2=tuple(int(x) for x in self._hsv_lo2) if self._hsv_dual and self._hsv_lo2 else None,
            box_hsv_upper2=tuple(int(x) for x in self._hsv_hi2) if self._hsv_dual and self._hsv_hi2 else None,
            box_color_min_pixels=max(1, int(self._color_min_px)),
            box_color_balloon_mode=True,
            box_color_log_interval_sec=0.28,
        )

    def _maybe_autostart_camera(self) -> None:
        try:
            if self.cap is None:
                self._start_camera()
        except Exception as e:
            self._log(f"[CAM] autostart hata: {e}")

    def toggle_camera(self) -> None:
        if self.cap is not None:
            self._stop_camera()
            return
        self._start_camera()

    def _start_engine_async(self, cfg: TrackerConfig) -> None:
        # UI thread'de model yüklemeyin: buton "donmuş" gibi hissediliyor.
        self.engine = None
        self._engine_pending_cfg = cfg
        if self._engine_signals is None:
            self._engine_signals = _EngineInitSignals()
            self._engine_signals.status.connect(self._on_engine_status)
            self._engine_signals.done.connect(self._on_engine_ready)
            self._engine_signals.failed.connect(self._on_engine_failed)

        self.btn_start_cam.setEnabled(False)
        worker = _EngineInitWorker(cfg, self.cap, bool(self.cb_autotune.isChecked()), self._engine_signals)
        self._engine_thread_pool.start(worker)

    def _on_engine_status(self, msg: str) -> None:
        self._log(msg)
        self.lbl_status.setText(msg)
        QtWidgets.QApplication.processEvents()

    def _on_engine_ready(self, cfg: TrackerConfig) -> None:
        try:
            self._log(f"[MODEL] cfg={asdict(cfg)}")
            self.engine = TrackerEngine(cfg)
            note = getattr(self.engine, "model_load_note", None)
            if note:
                self._log(note)
            self._log("[TRACK] aktif")
            cam = self.cb_camera.currentText().strip() or "0"
            short = os.path.basename(cfg.model) if cfg.model else "?"
            self.lbl_status.setText(f"Kamera {cam} | {short} | tespit/takip")
        except Exception as e:
            self._log(f"[MODEL] başlatılamadı: {e}")
            self.engine = None
        finally:
            self.btn_start_cam.setEnabled(True)
            self._engine_pending_cfg = None

    def _on_engine_failed(self, err: str) -> None:
        self._log(f"[MODEL] hata: {err}")
        self.engine = None
        self.btn_start_cam.setEnabled(True)
        self._engine_pending_cfg = None

    def _start_camera(self) -> None:
        self.btn_start_cam.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        cam_idx = int(self.cb_camera.currentText().strip() or "0")
        self.cap, backend_name = self._open_capture(cam_idx)
        if self.cap is None:
            self._log(f"[CAM] açılamadı: {cam_idx}")
            self.lbl_status.setText("Kamera açılamadı")
            self.btn_start_cam.setEnabled(True)
            return

        ok0, first = self._warmup_read(self.cap)
        if not ok0:
            self._log(f"[CAM] açıldı ama frame gelmiyor: idx={cam_idx} backend={backend_name}")
            self.lbl_status.setText("Kamera açıldı ama görüntü yok (backend/index dene)")
            self.cap.release()
            self.cap = None
            self.btn_start_cam.setEnabled(True)
            return

        cfg = self._current_cfg()
        fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fw and fh:
            apply_auto_heuristics(cfg, fw, fh)

        self.btn_start_cam.setText("Kamerayı Durdur")
        self.lbl_status.setText(f"Kamera: {cam_idx} | canlı (model yükleniyor...)")
        self.last_frame_time = time.time()
        self.fps_smooth = 0.0
        self.timer.start(10)
        self._log(f"[CAM] başladı: idx={cam_idx} backend={backend_name}")
        # İlk kareyi hemen göster (model yüklenirken siyah ekran kalmasın)
        try:
            self.video.set_frame(
                flip_and_enhance(
                    first,
                    mode=cfg.illumination,
                    clahe_clip=float(cfg.clahe_clip),
                    clahe_tile=int(cfg.clahe_tile),
                    denoise_light=bool(cfg.illum_denoise),
                )
            )
        except Exception:
            pass

        self._start_engine_async(cfg)
        self.btn_start_cam.setEnabled(True)

    def _stop_camera(self) -> None:
        self.timer.stop()
        try:
            self._engine_thread_pool.clear()
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.engine = None
        self._engine_pending_cfg = None
        self.video.setText("Kamera kapalı")
        self.btn_start_cam.setText("Kamerayı Başlat")
        self.lbl_status.setText("Hazır")
        self.lbl_fps.setText("FPS: -")
        self._log("[CAM] durduruldu")

    def _on_tick(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._log("[CAM] frame okunamadı")
            self._stop_camera()
            return

        cfg = self._current_cfg()
        if self.engine is not None:
            try:
                self._sync_engine_color_from_ui()
                display = self.engine.process_frame(frame)
                for line in self.engine.take_new_track_logs():
                    self._log(line)
                    print(line, flush=True)
                for line in self.engine.take_color_focus_logs():
                    self._log(line)
                    print(line, flush=True)
            except Exception as e:
                self._log(f"[TRACK] hata: {e}")
                display = flip_and_enhance(
                    frame,
                    mode=cfg.illumination,
                    clahe_clip=float(cfg.clahe_clip),
                    clahe_tile=int(cfg.clahe_tile),
                    denoise_light=bool(cfg.illum_denoise),
                )
        else:
            display = flip_and_enhance(
                frame,
                mode=cfg.illumination,
                clahe_clip=float(cfg.clahe_clip),
                clahe_tile=int(cfg.clahe_tile),
                denoise_light=bool(cfg.illum_denoise),
            )

        now = time.time()
        dt = max(1e-6, now - self.last_frame_time)
        self.last_frame_time = now
        fps = 1.0 / dt
        self.fps_smooth = fps if self.fps_smooth == 0.0 else (0.9 * self.fps_smooth + 0.1 * fps)
        self.lbl_fps.setText(f"FPS: {self.fps_smooth:.1f}")

        if self.btn_ff.isChecked():
            try:
                pf, pe, dom = analyze_friend_foe(display, self._ff_cfg)
                tnow = time.time()
                if (tnow - self._ff_last_log_ts) * 1000.0 >= float(self._ff_interval_ms):
                    self._ff_last_log_ts = tnow
                    line = f"[FF] dost={pf:.2f}% düşman={pe:.2f}% baskın={dom}"
                    self._log(line)
                    print(line, flush=True)
            except Exception as e:
                self._log(f"[FF] analiz hata: {e}")

        self.video.set_frame(display)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_camera()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


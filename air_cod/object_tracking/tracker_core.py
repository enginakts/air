from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Union

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from color_in_box import color_aim_in_bbox
from illumination import flip_and_enhance

# Ağırlık dosyaları burada aranır (IDE cwd fark etmez).
_TRACKER_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def default_local_model() -> str:
    """Önce `object_tracking` klasöründeki .pt dosyaları; yoksa hub adı `yolo12n.pt` (indirilebilir)."""
    for fn in (
        "yol12v.pt",
        "yolo12.pt",
        "yolo12s.pt",
        "yolo12n.pt",
        "yolo12m.pt",
        "yolo12l.pt",
        "yolo12x.pt",
        "yolov8n.pt",
    ):
        p = os.path.join(_TRACKER_PKG_DIR, fn)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return "yolo12n.pt"


def tracker_yaml_for_backend(backend: str) -> str:
    """Yerel `*_air.yaml` varsa kullan (GMC none / track_buffer↑); yoksa paket varsayılanı."""
    if backend == "botsort":
        p = os.path.join(_TRACKER_PKG_DIR, "botsort_air.yaml")
        return p if os.path.isfile(p) else "botsort.yaml"
    if backend == "bytetrack":
        p = os.path.join(_TRACKER_PKG_DIR, "bytetrack_air.yaml")
        return p if os.path.isfile(p) else "bytetrack.yaml"
    return ""


def resolve_model_path(model: str) -> str:
    """Tam yol / cwd / paket klasöründe `model` dosyasını çöz; bulunamazsa `model` stringini döndür (Ultralytics indirir)."""
    if not model:
        return default_local_model()
    if os.path.isfile(model):
        return os.path.abspath(model)
    base = os.path.basename(model)
    sibling = os.path.join(_TRACKER_PKG_DIR, base)
    if os.path.isfile(sibling):
        return os.path.abspath(sibling)
    return model


def _weights_is_yolo12_family(path: str) -> bool:
    b = os.path.basename(path).lower()
    return "yolo12" in b or "yol12" in b


def _is_ultralytics_arch_unpickle_error(exc: BaseException) -> bool:
    """Eski ultralytics ile YOLO11/12 .pt açılırken tipik hata (C3k2 vb. sınıf yok)."""
    t = str(exc).lower()
    return "c3k2" in t or "c2psa" in t or "a2c2f" in t or "can't get attribute" in t


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int


@dataclass
class TrackerConfig:
    # Ağırlıklar: `air_cod/object_tracking/` içine `yol12v.pt` / `yolo12.pt` koy; yoksa `yolov8n.pt` (indirilebilir).
    model: str = field(default_factory=default_local_model)
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    tile: int = 1
    min_box_area: int = 0
    max_age: int = 50
    n_init: int = 4
    flip_y: bool = False
    # Varsayılan BoT-SORT: Ultralytics yerleşik takip + proje içi `botsort_air.yaml` (sabit kamera, GMC kapalı=FPS).
    backend: Literal["deepsort", "bytetrack", "botsort"] = "botsort"
    # DeepSORT: her karede YOLO çalıştırmak yerine N karede bir çalıştır; arada Kalman öngörüsü (FPS↑, küçük gecikme).
    infer_every: int = 1
    max_det: int = 100
    embedder_gpu: Optional[bool] = None  # None: CUDA varsa True
    # DeepSORT ince ayar (hızlı el hareketi / ID kopması için gevşetilmiş varsayılanlar)
    deepsort_embedder: str = "mobilenet"  # mobilenet | clip_ViT-B/32 (CLIP için GPU şart)
    max_iou_distance: float = 0.88
    max_cosine_distance: float = 0.38
    nms_max_overlap: float = 0.55
    gating_only_position: bool = True
    # Işık / kontrast: LAB+CLAHE (+ otomatik gama). Literatürde düşük ışık YOLO ön işlemi için sık kullanılır.
    illumination: Literal["off", "auto", "low", "bright"] = "auto"
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    illum_denoise: bool = True
    # Aktif tespit kutusu içinde HSV — balon/hedef renk nişangâhı (Çelik Kubbe yarışması senaryosu).
    box_color_focus_enabled: bool = False
    box_hsv_lower: Tuple[int, int, int] = (0, 100, 100)
    box_hsv_upper: Tuple[int, int, int] = (10, 255, 255)
    # Tek inRange ile kapatılmayan hue (ör. kırmızı: düşük + yüksek H için ikinci çift).
    box_hsv_lower2: Optional[Tuple[int, int, int]] = None
    box_hsv_upper2: Optional[Tuple[int, int, int]] = None
    box_color_min_pixels: int = 80
    # Balon modu: morfoloji + en büyük kontur merkezi (hareket/durağan için daha stabil nişangâh).
    box_color_balloon_mode: bool = True
    box_color_log_interval_sec: float = 0.28


def yolo_detections(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    *,
    half: bool,
    device: Union[int, str, None],
    max_det: int,
) -> List[Detection]:
    results = model.predict(
        source=frame_bgr,
        verbose=False,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        half=half,
        max_det=max_det,
    )
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    boxes = r0.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    dets: List[Detection] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        dets.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(c), int(k)))
    return dets


def yolo_detections_tiled(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    tile_n: int,
    *,
    half: bool,
    device: Union[int, str, None],
    max_det: int,
) -> List[Detection]:
    if tile_n <= 1:
        return yolo_detections(
            model, frame_bgr, conf=conf, iou=iou, imgsz=imgsz, half=half, device=device, max_det=max_det
        )

    h, w = frame_bgr.shape[:2]
    xs = np.linspace(0, w, tile_n + 1, dtype=int)
    ys = np.linspace(0, h, tile_n + 1, dtype=int)

    out: List[Detection] = []
    for yi in range(tile_n):
        for xi in range(tile_n):
            x0, x1 = int(xs[xi]), int(xs[xi + 1])
            y0, y1 = int(ys[yi]), int(ys[yi + 1])
            tile = frame_bgr[y0:y1, x0:x1]
            dets = yolo_detections(
                model, tile, conf=conf, iou=iou, imgsz=imgsz, half=half, device=device, max_det=max_det
            )
            for d in dets:
                bx1, by1, bx2, by2 = d.xyxy
                out.append(Detection((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0), d.conf, d.cls))
    return out


def filter_detections(dets: List[Detection], min_box_area: int) -> List[Detection]:
    if min_box_area <= 0:
        return dets
    out: List[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area >= float(min_box_area):
            out.append(d)
    return out


def draw_raw_detections(
    frame: np.ndarray,
    dets: List[Detection],
    class_names: Dict[int, str],
    class_filter: Optional[List[int]] = None,
) -> None:
    """YOLO tespit kutuları (DeepSORT onayından bağımsız); kalın çerçeve + koyu kontur (Qt ölçekte kaybolmasın)."""
    for d in dets:
        if class_filter is not None and int(d.cls) not in class_filter:
            continue
        x1, y1, x2, y2 = int(d.xyxy[0]), int(d.xyxy[1]), int(d.xyxy[2]), int(d.xyxy[3])
        name = class_names.get(int(d.cls), str(int(d.cls)))
        # BGR: parlak camgöbeği; önce siyah kontur
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        lab = f"{d.conf:.2f} {name}"
        ty = max(18, y1 - 4)
        cv2.putText(frame, lab, (x1 + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, lab, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)


def to_deepsort_format(
    dets: List[Detection], class_filter: Optional[List[int]]
) -> List[Tuple[List[float], float, int]]:
    out: List[Tuple[List[float], float, int]] = []
    for d in dets:
        if class_filter is not None and d.cls not in class_filter:
            continue
        x1, y1, x2, y2 = d.xyxy
        out.append(([x1, y1, x2 - x1, y2 - y1], d.conf, d.cls))  # [x,y,w,h], conf, class
    return out


def draw_tracks(frame: np.ndarray, tracks, class_names: Dict[int, str]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.to_ltrb()]
        if not t.is_confirmed():
            # Henüz n_init kare dolmadı: turuncu kutu (kalın).
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                frame,
                "~",
                (x1 + 2, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            continue
        track_id = t.track_id
        cls = getattr(t, "det_class", None)
        cls_name = class_names.get(int(cls), str(cls)) if cls is not None else "?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"ID {track_id} | {cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 220, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def apply_auto_heuristics(cfg: TrackerConfig, frame_w: int, frame_h: int) -> None:
    cfg.conf = min(cfg.conf, 0.25)
    cfg.iou = max(cfg.iou, 0.55)

    if max(frame_w, frame_h) >= 1920:
        cfg.imgsz = max(cfg.imgsz, 960)
        cfg.tile = max(cfg.tile, 2)
        cfg.min_box_area = max(cfg.min_box_area, 12 * 12)
    elif max(frame_w, frame_h) >= 1280:
        cfg.imgsz = max(cfg.imgsz, 768)
        cfg.min_box_area = max(cfg.min_box_area, 10 * 10)
    else:
        cfg.imgsz = max(cfg.imgsz, 640)
        cfg.min_box_area = max(cfg.min_box_area, 8 * 8)


def _bench_one_config(
    model: YOLO,
    frames: List[np.ndarray],
    conf: float,
    iou: float,
    imgsz: int,
    tile: int,
    min_box_area: int,
    *,
    half: bool,
    device: Union[int, str, None],
    max_det: int,
) -> Tuple[float, float]:
    if not frames:
        return 0.0, 0.0
    t0 = time.time()
    det_sum = 0
    for f in frames:
        dets = yolo_detections_tiled(
            model, f, conf=conf, iou=iou, imgsz=imgsz, tile_n=tile, half=half, device=device, max_det=max_det
        )
        dets = filter_detections(dets, min_box_area=min_box_area)
        det_sum += len(dets)
    dt = max(1e-6, time.time() - t0)
    fps = float(len(frames)) / dt
    avg_det = float(det_sum) / float(len(frames))
    return fps, avg_det


def autotune_config(
    cfg: TrackerConfig,
    model: YOLO,
    cap: cv2.VideoCapture,
    tune_seconds: float = 4.0,
    target_fps: float = 18.0,
) -> TrackerConfig:
    frames: List[np.ndarray] = []
    t_end = time.time() + max(0.5, float(tune_seconds))
    while time.time() < t_end and len(frames) < 16:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if not frames:
        return cfg

    fw, fh = frames[0].shape[1], frames[0].shape[0]
    apply_auto_heuristics(cfg, fw, fh)

    use_cuda = _cuda_available()
    half = bool(use_cuda)
    device: Union[int, str, None] = 0 if use_cuda else "cpu"

    imgsz_candidates = sorted({640, 768, 960, int(cfg.imgsz)})
    tile_candidates = sorted({1, int(cfg.tile), 2})
    conf_candidates = sorted({0.20, 0.25, float(cfg.conf)})

    best = None
    best_meets = None
    for imgsz in imgsz_candidates:
        for tile in tile_candidates:
            for conf in conf_candidates:
                fps, avg_det = _bench_one_config(
                    model=model,
                    frames=frames,
                    conf=float(conf),
                    iou=float(cfg.iou),
                    imgsz=int(imgsz),
                    tile=int(tile),
                    min_box_area=int(cfg.min_box_area),
                    half=half,
                    device=device,
                    max_det=int(cfg.max_det),
                )
                cand = (avg_det, fps, imgsz, tile, conf)
                if fps >= float(target_fps):
                    if best_meets is None or cand > best_meets:
                        best_meets = cand
                if best is None or (fps, avg_det) > (best[1], best[0]):
                    best = cand

    chosen = best_meets if best_meets is not None else best
    if chosen is None:
        return cfg

    _, _, imgsz, tile, conf = chosen
    cfg.imgsz = int(imgsz)
    cfg.tile = int(tile)
    cfg.conf = float(conf)
    return cfg


class TrackerEngine:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        resolved = resolve_model_path(cfg.model)
        self.cfg.model = resolved
        self.model_load_note: Optional[str] = None
        try:
            self.model = YOLO(resolved)
        except Exception as e:
            if _weights_is_yolo12_family(resolved) and _is_ultralytics_arch_unpickle_error(e):
                fb = resolve_model_path("yolov8n.pt")
                self.cfg.model = fb
                self.model = YOLO(fb)
                self.model_load_note = (
                    "[MODEL] YOLO12 bu ultralytics sürümünde yüklenemedi (C3k2 vb.); "
                    f"geçici olarak {os.path.basename(fb)} açıldı. Kalıcı: aynı python ile "
                    "`pip install -U \"ultralytics>=8.3.0\"` — sonra uygulamayı yeniden başlat."
                )
            else:
                raise
        self._use_cuda = _cuda_available()
        self._half = bool(self._use_cuda)
        self._device: Union[int, str, None] = 0 if self._use_cuda else "cpu"
        self._frame_counter = 0
        self.class_names: Dict[int, str] = {}
        try:
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
            elif hasattr(self.model, "names") and isinstance(self.model.names, list):
                self.class_names = {i: str(n) for i, n in enumerate(self.model.names)}
        except Exception:
            self.class_names = {}

        self.tracker = None
        # Son YOLO çıkışı: infer_every>1 olsa bile her karede çizilir (kaybolan kutu hissi olmasın).
        self._last_overlay_dets: List[Detection] = []
        # Oturumda ilk kez görülen takip ID'leri — terminal/UI log için
        self._seen_track_ids: set[int] = set()
        self._new_track_logs: List[str] = []
        self._color_focus_logs: List[str] = []
        self._color_log_last_ts: Dict[int, float] = {}
        if cfg.backend == "deepsort":
            egpu = bool(cfg.embedder_gpu) if cfg.embedder_gpu is not None else bool(self._use_cuda)
            emb = (cfg.deepsort_embedder or "mobilenet").strip()
            valid_emb = (
                "mobilenet",
                "torchreid",
                "clip_RN50",
                "clip_RN101",
                "clip_RN50x4",
                "clip_RN50x16",
                "clip_ViT-B/32",
                "clip_ViT-B/16",
            )
            if emb not in valid_emb:
                emb = "mobilenet"
            if emb != "mobilenet" and not self._use_cuda:
                emb = "mobilenet"
            # half: mobilenet için CUDA’da FP16; CLIP tarafı kütüphane içinde yönetilir.
            self.tracker = DeepSort(
                max_age=int(cfg.max_age),
                n_init=int(cfg.n_init),
                nms_max_overlap=float(cfg.nms_max_overlap),
                max_iou_distance=float(cfg.max_iou_distance),
                max_cosine_distance=float(cfg.max_cosine_distance),
                nn_budget=120,
                gating_only_position=bool(cfg.gating_only_position),
                embedder=emb,
                half=bool(self._use_cuda),
                bgr=True,
                embedder_gpu=egpu,
            )

        # Ultralytics trackers keep state when persist=True (yerel yaml tam yol / ileri eğik çizgi)
        if cfg.backend != "deepsort":
            yp = tracker_yaml_for_backend(cfg.backend)
            if yp and os.path.isfile(yp):
                self._ultra_tracker_name = os.path.normpath(yp).replace("\\", "/")
            else:
                self._ultra_tracker_name = yp
        else:
            self._ultra_tracker_name = ""

    def take_new_track_logs(self) -> List[str]:
        """Bu motor örneğinde ilk kez atanan iz ID'leri için biriken log satırlarını verir ve listeyi temizler."""
        out = self._new_track_logs
        self._new_track_logs = []
        return out

    def take_color_focus_logs(self) -> List[str]:
        out = self._color_focus_logs
        self._color_focus_logs = []
        return out

    def _apply_box_color_focus(
        self,
        work_bgr: np.ndarray,
        out_bgr: np.ndarray,
        tid: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        if not self.cfg.box_color_focus_enabled:
            return
        low = tuple(int(v) for v in self.cfg.box_hsv_lower)
        high = tuple(int(v) for v in self.cfg.box_hsv_upper)
        ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [(low, high)]
        lo2 = self.cfg.box_hsv_lower2
        hi2 = self.cfg.box_hsv_upper2
        if lo2 is not None and hi2 is not None:
            ranges.append((tuple(int(v) for v in lo2), tuple(int(v) for v in hi2)))
        minpx = max(1, int(self.cfg.box_color_min_pixels))
        aim = color_aim_in_bbox(
            work_bgr,
            x1,
            y1,
            x2,
            y2,
            ranges,
            minpx,
            balloon_mode=bool(self.cfg.box_color_balloon_mode),
        )
        if aim is None:
            return
        gx, gy, qual = aim
        fh, fw = work_bgr.shape[:2]
        nx = gx / max(1.0, float(fw))
        ny = gy / max(1.0, float(fh))
        ix, iy = int(round(gx)), int(round(gy))
        cyan = (255, 255, 0)  # BGR cyan (yüksek görünürlük)
        cv2.circle(out_bgr, (ix, iy), 8, cyan, 1, lineType=cv2.LINE_AA)
        cv2.drawMarker(
            out_bgr,
            (ix, iy),
            cyan,
            markerType=cv2.MARKER_CROSS,
            markerSize=26,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        now = time.time()
        gap = float(self.cfg.box_color_log_interval_sec)
        if now - self._color_log_last_ts.get(tid, 0.0) < gap:
            return
        self._color_log_last_ts[tid] = now
        self._color_focus_logs.append(
            f"[AIM] ID={tid} px=({gx:.2f},{gy:.2f}) norm=({nx:.4f},{ny:.4f}) "
            f"kutu=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) kalite={qual:.3f}"
        )

    def _log_new_track_if_first(
        self,
        tid: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cls_opt: Optional[Union[int, float]],
    ) -> None:
        if tid in self._seen_track_ids:
            return
        self._seen_track_ids.add(tid)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if cls_opt is not None:
            ci = int(cls_opt)
            cname = self.class_names.get(ci, str(ci))
        else:
            cname = "?"
        line = (
            f"[TRACK] yeni nesne ID={tid} merkez=({cx:.1f},{cy:.1f}) "
            f"kutu=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) sınıf={cname}"
        )
        self._new_track_logs.append(line)

    def process_frame(self, frame_bgr: np.ndarray, class_filter: Optional[List[int]] = None) -> np.ndarray:
        # Önce Y flip, sonra loş/parlak ortama göre kontrast (CLAHE LAB) — tespit/takip bu kareyi görür.
        work = flip_and_enhance(
            frame_bgr,
            mode=self.cfg.illumination,
            clahe_clip=float(self.cfg.clahe_clip),
            clahe_tile=int(self.cfg.clahe_tile),
            denoise_light=bool(self.cfg.illum_denoise),
        )
        out = work.copy()
        self._frame_counter += 1
        run_yolo = (self._frame_counter % max(1, int(self.cfg.infer_every))) == 0
        if self.cfg.backend == "deepsort":
            if run_yolo:
                dets = yolo_detections_tiled(
                    self.model,
                    work,
                    conf=self.cfg.conf,
                    iou=self.cfg.iou,
                    imgsz=self.cfg.imgsz,
                    tile_n=self.cfg.tile,
                    half=self._half,
                    device=self._device,
                    max_det=int(self.cfg.max_det),
                )
                dets = filter_detections(dets, min_box_area=self.cfg.min_box_area)
                dets.sort(key=lambda d: -float(d.conf))
                self._last_overlay_dets = dets
                ds_dets = to_deepsort_format(dets, class_filter)
            else:
                ds_dets = []
            tracks = self.tracker.update_tracks(ds_dets, frame=work) if self.tracker is not None else []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = int(t.track_id)
                x1, y1, x2, y2 = t.to_ltrb()
                dc = getattr(t, "det_class", None)
                self._log_new_track_if_first(tid, float(x1), float(y1), float(x2), float(y2), dc)
            # Her karede son YOLO kutuları (infer_every ile ara verilen karelerde de görünsün).
            draw_raw_detections(out, self._last_overlay_dets, self.class_names, class_filter)
            draw_tracks(out, tracks, self.class_names)
            if self.cfg.box_color_focus_enabled:
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    tid = int(t.track_id)
                    bx1, by1, bx2, by2 = t.to_ltrb()
                    self._apply_box_color_focus(work, out, tid, float(bx1), float(by1), float(bx2), float(by2))
        else:
            # Ultralytics built-in trackers (ByteTrack / BoT-SORT). These are Kalman-based trackers.
            results = self.model.track(
                source=work,
                persist=True,
                verbose=False,
                tracker=self._ultra_tracker_name,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                imgsz=self.cfg.imgsz,
                half=self._half,
                device=self._device,
                max_det=int(self.cfg.max_det),
            )
            if results:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    clss = r0.boxes.cls.cpu().numpy().astype(int)
                    ids = None
                    if getattr(r0.boxes, "id", None) is not None:
                        ids = r0.boxes.id.cpu().numpy().astype(int)
                    boxed: List[Tuple[float, float, float, float, int, int]] = []
                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        cls = int(clss[i])
                        if class_filter is not None and cls not in class_filter:
                            continue
                        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                        if self.cfg.min_box_area > 0 and area < float(self.cfg.min_box_area):
                            continue
                        tid = int(ids[i]) if ids is not None else -1
                        boxed.append((float(x1), float(y1), float(x2), float(y2), cls, tid))
                    for x1, y1, x2, y2, cls, tid in boxed:
                        if tid != -1:
                            self._log_new_track_if_first(tid, x1, y1, x2, y2, cls)
                        label = f"ID {tid} | {self.class_names.get(cls, str(cls))}" if tid != -1 else self.class_names.get(cls, str(cls))
                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))
                        cv2.rectangle(out, p1, p2, (0, 220, 0), 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(out, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), (0, 220, 0), -1)
                        cv2.putText(out, label, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    if self.cfg.box_color_focus_enabled:
                        for x1, y1, x2, y2, _cls, tid in boxed:
                            if tid == -1:
                                continue
                            self._apply_box_color_focus(work, out, tid, x1, y1, x2, y2)

        return out


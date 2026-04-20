import argparse
import time
from typing import List, Optional, Union

import cv2

from tracker_core import (
    TrackerConfig,
    TrackerEngine,
    apply_auto_heuristics,
    autotune_config,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + DeepSORT (Kalman) object detection & tracking")
    p.add_argument("--source", default="0", help='Video kaynağı: "0" webcam, veya video yolu/URL')
    p.add_argument(
        "--model",
        default="yolo12n.pt",
        help="YOLO model yolu/adı (örn: yolo12n.pt, yol12v.pt, yolov8n.pt). İndirmek için: python download_yolo12_weights.py",
    )
    p.add_argument(
        "--backend",
        default="botsort",
        choices=["deepsort", "bytetrack", "botsort"],
        help="Takip: botsort (önerilen, proje botsort_air.yaml), bytetrack, deepsort (REID+Kalman)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--classes", default="", help='Sadece bu class id\'ler (örn: "0,2,3"). Boş: hepsi')
    p.add_argument("--auto", action="store_true", help="Sisteme/kameraya göre otomatik ayar seç (imgsz/conf/tile)")
    p.add_argument(
        "--autotune",
        action="store_true",
        help="Başlangıçta kısa benchmark ile en iyi ayarı seç (imgsz/conf/tile)",
    )
    p.add_argument("--tune-seconds", type=float, default=4.0, help="Autotune süresi (sn)")
    p.add_argument("--target-fps", type=float, default=18.0, help="Autotune hedef FPS (altına düşmemeye çalışır)")
    p.add_argument(
        "--tile",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Küçük/uzak hedefler için görüntüyü NxN bölerek tespit (1=kapalı, 2=2x2, 3=3x3)",
    )
    p.add_argument("--min-box-area", type=int, default=0, help="Bu alandan (px^2) küçük bbox'ları at")
    p.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="DeepSORT: YOLO'yu her N karede bir çalıştır (1=her kare, 2≈2x FPS; arada Kalman öngörüsü)",
    )
    p.add_argument("--max-det", type=int, default=100, help="YOLO max tespit sayısı (düşük=FPS↑)")
    p.add_argument(
        "--illumination",
        default="auto",
        choices=["off", "auto", "low", "bright"],
        help="Işık ön işleme: auto=parlaklığa göre LAB+CLAHE; low=loş zorunlu; bright=aşırı parlak",
    )
    p.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (üst sınır)")
    p.add_argument("--clahe-tile", type=int, default=8, help="CLAHE tile grid (2-32)")
    p.add_argument(
        "--no-illum-denoise",
        action="store_true",
        help="Çok karanlıkta kullanılan hafif Gaussian ön filtreyi kapat (biraz daha keskin, daha gürültülü)",
    )
    p.add_argument("--max-age", type=int, default=30, help="DeepSORT track max age")
    p.add_argument("--n-init", type=int, default=3, help="DeepSORT track init hits")
    p.add_argument("--window", default="Tracking", help="Pencere başlığı")
    p.add_argument("--show-fps", action="store_true", help="FPS yazdır")
    p.add_argument(
        "--flip-y",
        action="store_true",
        help="(Eski) Artık kullanılmıyor: Y çevirme her zaman tespit/takipten önce uygulanır",
    )
    p.add_argument("--save", default="", help="Çıktıyı kaydet (örn: out.mp4). Boş: kaydetme")
    return p.parse_args()


def parse_source(source: str) -> Union[int, str]:
    s = source.strip()
    if s.isdigit():
        return int(s)
    return s


def parse_classes(s: str) -> Optional[List[int]]:
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    args = parse_args()
    args.model = resolve_model_path(args.model)
    source = parse_source(args.source)
    class_filter = parse_classes(args.classes)
    cfg = TrackerConfig(
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tile=args.tile,
        min_box_area=args.min_box_area,
        infer_every=max(1, int(args.infer_every)),
        max_det=max(1, int(args.max_det)),
        max_age=args.max_age,
        n_init=args.n_init,
        flip_y=False,
        backend=args.backend,
        illumination=str(args.illumination),
        clahe_clip=float(args.clahe_clip),
        clahe_tile=int(args.clahe_tile),
        illum_denoise=not bool(args.no_illum_denoise),
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {args.source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if args.auto and fw > 0 and fh > 0:
        apply_auto_heuristics(cfg, fw, fh)

    engine = TrackerEngine(cfg)
    if args.autotune:
        cfg = autotune_config(cfg, engine.model, cap, tune_seconds=float(args.tune_seconds), target_fps=float(args.target_fps))
        engine = TrackerEngine(cfg)
        print(f"[autotune] selected imgsz={cfg.imgsz} tile={cfg.tile} conf={cfg.conf:.2f}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    last_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = engine.process_frame(frame, class_filter=class_filter)
            for line in engine.take_new_track_logs():
                print(line, flush=True)
            for line in engine.take_color_focus_logs():
                print(line, flush=True)

            if args.show_fps:
                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now
                fps = 1.0 / dt
                fps_smooth = fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * fps)
                cv2.putText(
                    frame,
                    f"FPS: {fps_smooth:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (30, 30, 255),
                    2,
                )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(args.window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


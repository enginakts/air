from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .config import AppConfig, TARGET_CLASS_NAMES
from .color_iff import classify_friend_enemy
from .detector import YoloDetector
from .tracker import DeepSortTracker
from .ui import OpenCvUi
from .mission import MissionConfig, MissionManager, MotionInfo
from .telemetry import NullSink, SerialTelemetrySink, TelemetryPacket, UdpTelemetrySink


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Celikkube object detection + tracking (YOLO + DeepSORT + OpenCV).")
    p.add_argument("--weights", required=True, help="Path to YOLO weights (.pt).")
    p.add_argument("--source", default=None, help="Video source: camera index (0) or path to video file.")
    p.add_argument("--select-camera", action="store_true", help="Interactive camera selection (ignores --source if set).")
    p.add_argument("--device", default=None, help='Ultralytics device, e.g. "cpu" or "cuda:0".')
    p.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")

    p.add_argument("--max-age", type=int, default=30, help="DeepSORT max_age.")
    p.add_argument("--n-init", type=int, default=3, help="DeepSORT n_init.")
    p.add_argument("--max-iou-distance", type=float, default=0.7, help="DeepSORT max_iou_distance.")
    p.add_argument("--no-kalman-smoothing", action="store_true", help="Disable extra Kalman smoothing layer.")

    # Telemetry outputs (simulation-safe)
    p.add_argument("--stm32-port", default=None, help='STM32 serial port for telemetry (e.g. "COM3").')
    p.add_argument("--stm32-baud", type=int, default=115200, help="STM32 serial baud rate.")
    p.add_argument("--jetson-udp", default=None, help='Jetson UDP telemetry target "host:port" (e.g. 192.168.1.50:5005).')

    return p.parse_args()


def _open_capture(source: str) -> cv2.VideoCapture:
    # camera index?
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> int:
    args = _parse_args()

    cfg = AppConfig(
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
        device=args.device,
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.max_iou_distance,
    )

    from .camera_select import pick_camera_interactive

    if args.select_camera:
        cam_idx = pick_camera_interactive(max_index=10)
        source = str(cam_idx)
    else:
        source = str(args.source) if args.source is not None else "0"

    cap = _open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Video source could not be opened: {source}")

    detector = YoloDetector(
        weights_path=str(args.weights),
        device=cfg.device,
        img_size=cfg.img_size,
        conf=cfg.conf,
        iou=cfg.iou,
    )

    tracker = DeepSortTracker(
        max_age=cfg.max_age,
        n_init=cfg.n_init,
        max_iou_distance=cfg.max_iou_distance,
        use_kalman_smoothing=(not args.no_kalman_smoothing),
    )

    runs_dir = Path("runs")
    ui = OpenCvUi(cfg.window_name, runs_dir=runs_dir)
    mission = MissionManager()
    prev_center_by_id: dict[int, tuple[float, float, float]] = {}  # tid -> (cx,cy,ts)
    prev_dist_dm_by_id: dict[int, tuple[int, float]] = {}  # tid -> (dist_dm, ts)

    last_t = time.perf_counter()
    fps = 0.0

    # Telemetry sinks (no actuation / no fire command)
    stm32_sink = NullSink()
    if args.stm32_port:
        stm32_sink = SerialTelemetrySink(port=str(args.stm32_port), baud=int(args.stm32_baud))

    jetson_sink = NullSink()
    if args.jetson_udp:
        host, port_s = str(args.jetson_udp).split(":", 1)
        jetson_sink = UdpTelemetrySink(host=host, port=int(port_s))

    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("Could not read first frame.")

    while True:
        if not ui.state.paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        ui.set_last_frame(frame)

        # time / fps
        now = time.perf_counter()
        dt = max(1e-6, now - last_t)
        last_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # live thresholds (OpenCV trackbars)
        conf, iou = ui.get_thresholds()
        detector.conf = conf
        detector.iou = iou

        dets = detector.infer(frame)

        # Filter detections to 4 classes if model contains more
        dets = [d for d in dets if 0 <= d.cls < len(TARGET_CLASS_NAMES)]

        tracks = tracker.update(frame, dets)

        # IFF (friend/enemy by user-picked colors)
        fr_tol, en_tol, min_ratio = ui.get_iff_settings()
        iff_by_id: dict[int, tuple[str, float]] = {}
        for tr in tracks:
            lab, score = classify_friend_enemy(
                frame_bgr=frame,
                box_xyxy=(tr.x1, tr.y1, tr.x2, tr.y2),
                friend=ui.state.friend_color,
                enemy=ui.state.enemy_color,
                friend_tol=fr_tol,
                enemy_tol=en_tol,
                min_ratio=min_ratio,
            )
            iff_by_id[tr.track_id] = (lab, score)

        # Range gating inputs + distance estimate (monocular proxy using bbox width)
        ref_dist_dm, ref_bbox_w_px, ranges_dm = ui.get_range_settings()
        ui.state.ref_dist_dm = ref_dist_dm
        dist_by_id: dict[int, float] = {}
        if ref_bbox_w_px > 0 and ref_dist_dm > 0:
            ref_dist_m = ref_dist_dm / 10.0
            for tr in tracks:
                bw = max(1, tr.x2 - tr.x1)
                dist_m = ref_dist_m * (ref_bbox_w_px / float(bw))
                dist_by_id[tr.track_id] = float(dist_m)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Motion estimation (direction/speed) per track id
        motion_by_id: dict[int, MotionInfo] = {}
        live_ids = {t.track_id for t in tracks}
        for tid in list(prev_center_by_id.keys()):
            if tid not in live_ids:
                prev_center_by_id.pop(tid, None)
        for tid in list(prev_dist_dm_by_id.keys()):
            if tid not in live_ids:
                prev_dist_dm_by_id.pop(tid, None)

        for tr in tracks:
            vx = vy = 0.0
            if tr.track_id in prev_center_by_id:
                pcx, pcy, pts = prev_center_by_id[tr.track_id]
                dtm = max(1e-3, now - pts)
                vx = (tr.cx - pcx) / dtm
                vy = (tr.cy - pcy) / dtm
            prev_center_by_id[tr.track_id] = (tr.cx, tr.cy, now)

            dist_rate = 0.0
            if tr.track_id in dist_by_id:
                d_dm = int(round(dist_by_id[tr.track_id] * 10.0))
                if tr.track_id in prev_dist_dm_by_id:
                    pdm, pts = prev_dist_dm_by_id[tr.track_id]
                    dtm = max(1e-3, now - pts)
                    dist_rate = (d_dm - pdm) / dtm
                prev_dist_dm_by_id[tr.track_id] = (d_dm, now)

            motion_by_id[tr.track_id] = MotionInfo(vx_px_s=float(vx), vy_px_s=float(vy), dist_rate_dm_s=float(dist_rate))

        # Mission settings + stage selection (simulation only)
        stage, mcfg = ui.get_mission_settings()
        mission.set_stage(stage)

        evt = mission.step(
            now=now,
            engage_enabled=ui.state.engagement_enabled,
            crosshair_xy=(cx, cy),
            tracks=tracks,
            iff_by_id=iff_by_id,
            dist_by_id=dist_by_id,
            motion_by_id=motion_by_id,
            cfg=mcfg,
        )

        # Telemetry: send one packet per track (best-effort)
        ev_kind = evt.kind if evt is not None else ""
        ev_label = evt.label if evt is not None else ""
        for tr in tracks:
            lab, _ = iff_by_id.get(tr.track_id, ("unknown", 0.0))
            mi = motion_by_id.get(tr.track_id, MotionInfo(0.0, 0.0, 0.0))
            pkt = TelemetryPacket(
                ts=now,
                stage=mission.state.stage,
                track_id=tr.track_id,
                cls=int(tr.cls),
                iff=lab,
                conf=float(tr.conf),
                bbox=(int(tr.x1), int(tr.y1), int(tr.x2), int(tr.y2)),
                center=(float(tr.cx), float(tr.cy)),
                vel_px_s=(float(mi.vx_px_s), float(mi.vy_px_s)),
                dist_m=float(dist_by_id.get(tr.track_id, 0.0)),
                event=(ev_kind if (evt is not None and evt.track_id == tr.track_id) else ""),
                event_label=(ev_label if (evt is not None and evt.track_id == tr.track_id) else ""),
            )
            stm32_sink.send(pkt)
            jetson_sink.send(pkt)

        vis = ui.draw(
            frame,
            tracks,
            fps=fps,
            iff_by_id=iff_by_id,
            fired_track_id=(evt.track_id if (evt is not None and evt.kind in {"virtual_hit", "stage_complete"} and ui.state.engagement_enabled) else None),
            dist_by_id=dist_by_id,
            motion_by_id=motion_by_id,
            mission_state=mission.state,
            mission_event=(None if evt is None else (evt.kind, evt.label)),
        )

        # recording
        ui.set_recording(ui.state.recording, vis, fps=fps)
        ui.write_record_frame(vis)

        ui.show(vis)
        key = cv2.waitKey(1) & 0xFF
        action = ui.handle_key(key)
        if action == "quit":
            break
        if action == "snapshot":
            ui.snapshot(vis)
        if action == "pick_ref_bbox":
            # Pick a reference bbox width at the currently selected ref distance.
            # Prefer enemy under crosshair, else largest confirmed track.
            hh, ww = frame.shape[:2]
            cx, cy = ww // 2, hh // 2
            picked = None
            for tr in tracks:
                lab, _ = iff_by_id.get(tr.track_id, ("unknown", 0.0))
                if lab != "enemy":
                    continue
                if tr.x1 <= cx <= tr.x2 and tr.y1 <= cy <= tr.y2:
                    picked = tr
                    break
            if picked is None and tracks:
                picked = max(tracks, key=lambda t: (t.x2 - t.x1) * (t.y2 - t.y1))
            if picked is not None:
                ui.state.ref_bbox_w_px = max(1, picked.x2 - picked.x1)

    ui.close()
    cap.release()
    try:
        stm32_sink.close()
        jetson_sink.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


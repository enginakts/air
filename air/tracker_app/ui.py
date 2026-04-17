from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .config import TARGET_CLASS_NAMES
from .color_iff import HsvColor, HsvTol, bgr_to_hsv
from .tracker import Track
from .mission import MissionConfig, MissionState, MotionInfo


@dataclass
class UiState:
    paused: bool = False
    recording: bool = False
    engagement_enabled: bool = False
    selected_classes: Set[int] = None  # type: ignore[assignment]
    friend_color: Optional[HsvColor] = None
    enemy_color: Optional[HsvColor] = None
    pick_mode: str = "friend"  # "friend" | "enemy"
    range_gate_enabled: bool = False
    ref_dist_dm: int = 100  # decimeters, default 10.0m
    ref_bbox_w_px: int = 0  # calibrated pixel width at ref_dist

    def __post_init__(self) -> None:
        if self.selected_classes is None:
            self.selected_classes = set(range(len(TARGET_CLASS_NAMES)))


class OpenCvUi:
    def __init__(self, window_name: str, runs_dir: Path) -> None:
        self.window_name = window_name
        self.runs_dir = runs_dir
        self.state = UiState()

        self._writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[Path] = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._init_trackbars()
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self._last_frame: Optional[np.ndarray] = None

    def _init_trackbars(self) -> None:
        def _noop(_: int) -> None:
            return

        # 0..100 -> 0.00..1.00
        cv2.createTrackbar("conf x100", self.window_name, 35, 100, _noop)
        cv2.createTrackbar("iou  x100", self.window_name, 45, 100, _noop)

        # IFF (friend/enemy by color in HSV)
        cv2.createTrackbar("min color% ", self.window_name, 8, 50, _noop)  # 0..50%

        cv2.createTrackbar("FR dh", self.window_name, 10, 90, _noop)
        cv2.createTrackbar("FR ds", self.window_name, 60, 255, _noop)
        cv2.createTrackbar("FR dv", self.window_name, 60, 255, _noop)

        cv2.createTrackbar("EN dh", self.window_name, 10, 90, _noop)
        cv2.createTrackbar("EN ds", self.window_name, 60, 255, _noop)
        cv2.createTrackbar("EN dv", self.window_name, 60, 255, _noop)

        # Range gating (5..15m) per class, in decimeters to keep integers
        cv2.createTrackbar("ref dist dm", self.window_name, 100, 300, _noop)  # allow up to 30m
        # R? min/max: 0 means "unset"
        for i in range(min(4, len(TARGET_CLASS_NAMES))):
            cv2.createTrackbar(f"R{i} min dm", self.window_name, 0, 150, _noop)
            cv2.createTrackbar(f"R{i} max dm", self.window_name, 0, 150, _noop)

        # Mission / stages (simulation)
        cv2.createTrackbar("stage 1-3", self.window_name, 1, 3, _noop)
        cv2.createTrackbar("S1 tol dm", self.window_name, 10, 50, _noop)
        cv2.createTrackbar("S1 step0 cls", self.window_name, 0, 3, _noop)
        cv2.createTrackbar("S1 step1 cls", self.window_name, 1, 3, _noop)
        cv2.createTrackbar("S1 step2 cls", self.window_name, 2, 3, _noop)
        cv2.createTrackbar("S2 want dm", self.window_name, 100, 150, _noop)
        cv2.createTrackbar("S2 tol dm", self.window_name, 10, 50, _noop)

    def get_thresholds(self) -> Tuple[float, float]:
        conf100 = cv2.getTrackbarPos("conf x100", self.window_name)
        iou100 = cv2.getTrackbarPos("iou  x100", self.window_name)
        conf = max(0.0, min(1.0, conf100 / 100.0))
        iou = max(0.0, min(1.0, iou100 / 100.0))
        return conf, iou

    def get_iff_settings(self) -> Tuple[HsvTol, HsvTol, float]:
        min_pct = cv2.getTrackbarPos("min color% ", self.window_name)
        min_ratio = max(0.0, min(0.5, min_pct / 100.0))

        fr = HsvTol(
            dh=cv2.getTrackbarPos("FR dh", self.window_name),
            ds=cv2.getTrackbarPos("FR ds", self.window_name),
            dv=cv2.getTrackbarPos("FR dv", self.window_name),
        )
        en = HsvTol(
            dh=cv2.getTrackbarPos("EN dh", self.window_name),
            ds=cv2.getTrackbarPos("EN ds", self.window_name),
            dv=cv2.getTrackbarPos("EN dv", self.window_name),
        )
        return fr, en, min_ratio

    def get_range_settings(self) -> Tuple[int, int, list[tuple[int, int]]]:
        ref_dist_dm = cv2.getTrackbarPos("ref dist dm", self.window_name)
        ranges: list[tuple[int, int]] = []
        for i in range(min(4, len(TARGET_CLASS_NAMES))):
            mn = cv2.getTrackbarPos(f"R{i} min dm", self.window_name)
            mx = cv2.getTrackbarPos(f"R{i} max dm", self.window_name)
            ranges.append((mn, mx))
        return ref_dist_dm, self.state.ref_bbox_w_px, ranges

    def get_mission_settings(self) -> tuple[int, MissionConfig]:
        stage = cv2.getTrackbarPos("stage 1-3", self.window_name)
        s1_tol = cv2.getTrackbarPos("S1 tol dm", self.window_name)
        mcfg = MissionConfig(
            s1_cls_step0=cv2.getTrackbarPos("S1 step0 cls", self.window_name),
            s1_cls_step1=cv2.getTrackbarPos("S1 step1 cls", self.window_name),
            s1_cls_step2=cv2.getTrackbarPos("S1 step2 cls", self.window_name),
            s1_dist_dm_step0=50,
            s1_dist_dm_step1=100,
            s1_dist_dm_step2=150,
            s1_tol_dm=s1_tol,
            s2_desired_dm=cv2.getTrackbarPos("S2 want dm", self.window_name),
            s2_tol_dm=cv2.getTrackbarPos("S2 tol dm", self.window_name),
        )
        return stage, mcfg

    def handle_key(self, key: int) -> str:
        # returns action: "quit" | "toggle_pause" | "toggle_record" | "snapshot" | ""
        if key in (ord("q"), 27):  # q or ESC
            return "quit"
        if key == ord(" "):
            self.state.paused = not self.state.paused
            return "toggle_pause"
        if key == ord("r"):
            self.state.recording = not self.state.recording
            return "toggle_record"
        if key == ord("s"):
            return "snapshot"
        if key == ord("t"):
            self.state.engagement_enabled = not self.state.engagement_enabled
            return ""
        if key == ord("f"):
            self.state.pick_mode = "friend"
            return ""
        if key == ord("e"):
            self.state.pick_mode = "enemy"
            return ""
        if key == ord("c"):
            self.state.friend_color = None
            self.state.enemy_color = None
            return ""
        if key == ord("g"):
            self.state.range_gate_enabled = not self.state.range_gate_enabled
            return ""
        if key == ord("p"):
            # request a ref bbox pick (handled in main when it knows tracks)
            return "pick_ref_bbox"
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            cls_id = key - ord("1")
            if cls_id in self.state.selected_classes:
                self.state.selected_classes.remove(cls_id)
            else:
                self.state.selected_classes.add(cls_id)
            return ""
        return ""

    def set_last_frame(self, frame_bgr: np.ndarray) -> None:
        self._last_frame = frame_bgr

    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._last_frame is None:
            return
        h, w = self._last_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        b, g, r = [int(v) for v in self._last_frame[y, x]]
        hsv = bgr_to_hsv((b, g, r))
        if self.state.pick_mode == "friend":
            self.state.friend_color = hsv
        else:
            self.state.enemy_color = hsv

    def draw(
        self,
        frame_bgr: np.ndarray,
        tracks: Sequence[Track],
        fps: float,
        iff_by_id: Optional[dict[int, tuple[str, float]]] = None,
        fired_track_id: Optional[int] = None,
        dist_by_id: Optional[dict[int, float]] = None,
        motion_by_id: Optional[dict[int, MotionInfo]] = None,
        mission_state: Optional[MissionState] = None,
        mission_event: Optional[tuple[str, str]] = None,
    ) -> np.ndarray:
        img = frame_bgr.copy()

        # Draw tracks
        for tr in tracks:
            if tr.cls != -1 and tr.cls not in self.state.selected_classes:
                continue
            color = _color_for_id(tr.track_id)
            cv2.rectangle(img, (tr.x1, tr.y1), (tr.x2, tr.y2), color, 2)

            name = TARGET_CLASS_NAMES[tr.cls] if 0 <= tr.cls < len(TARGET_CLASS_NAMES) else "unknown"
            iff_txt = ""
            if iff_by_id is not None and tr.track_id in iff_by_id:
                lab, score = iff_by_id[tr.track_id]
                iff_txt = f" | {lab}:{score:.2f}"

            dist_txt = ""
            if dist_by_id is not None and tr.track_id in dist_by_id:
                dist_txt = f" | {dist_by_id[tr.track_id]:.1f}m"

            label = f"ID {tr.track_id} | {name} | {tr.conf:.2f}{iff_txt}{dist_txt}"
            _put_label(img, label, tr.x1, tr.y1 - 6, color)

            cv2.circle(img, (int(tr.cx), int(tr.cy)), 3, color, -1)

            # Motion arrow (direction of travel)
            if motion_by_id is not None and tr.track_id in motion_by_id:
                mi = motion_by_id[tr.track_id]
                # scale velocity to a short arrow
                ax = int(tr.cx + float(np.clip(mi.vx_px_s, -500.0, 500.0)) * 0.05)
                ay = int(tr.cy + float(np.clip(mi.vy_px_s, -500.0, 500.0)) * 0.05)
                cv2.arrowedLine(img, (int(tr.cx), int(tr.cy)), (ax, ay), color, 2, tipLength=0.3)

            if fired_track_id is not None and tr.track_id == fired_track_id:
                cv2.putText(
                    img,
                    "FIRE",
                    (tr.x1, tr.y2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        # crosshair
        hh, ww = img.shape[:2]
        cx, cy = ww // 2, hh // 2
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)

        # HUD
        hud1 = (
            f"FPS: {fps:.1f}  |  Pause: {'ON' if self.state.paused else 'OFF'}"
            f"  |  Rec: {'ON' if self.state.recording else 'OFF'}"
            f"  |  Engage(t): {'ON' if self.state.engagement_enabled else 'OFF'}"
        )
        cv2.putText(img, hud1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 240, 20), 2, cv2.LINE_AA)

        cls_flags = []
        for i, n in enumerate(TARGET_CLASS_NAMES[:4]):
            on = i in self.state.selected_classes
            cls_flags.append(f"{i+1}:{n}={'ON' if on else 'OFF'}")
        hud2 = "  ".join(cls_flags)
        cv2.putText(img, hud2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        fr = "set" if self.state.friend_color is not None else "none"
        en = "set" if self.state.enemy_color is not None else "none"
        hud3 = f"Pick: (f/e)={self.state.pick_mode} | Friend={fr} Enemy={en} | Click to sample | (c) clear"
        cv2.putText(img, hud3, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

        rg = "ON" if self.state.range_gate_enabled else "OFF"
        refw = self.state.ref_bbox_w_px if self.state.ref_bbox_w_px > 0 else 0
        hud4 = f"RangeGate(g): {rg} | Calib(p): ref_bbox_w_px={refw} | ref dist dm trackbar"
        cv2.putText(img, hud4, (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

        if mission_state is not None:
            hud5 = f"Stage={mission_state.stage} | Hits={mission_state.hits} Miss={mission_state.misses} FriendViol={mission_state.friend_violations} | S1 step={mission_state.s1_step}/3"
            cv2.putText(img, hud5, (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 40), 2, cv2.LINE_AA)

        if mission_event is not None:
            kind, label = mission_event
            cv2.putText(img, f"EVENT: {kind} ({label})", (12, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2, cv2.LINE_AA)

        if self._record_path is not None:
            cv2.putText(
                img,
                f"REC -> {self._record_path.name}",
                (12, 196),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return img

    def show(self, img_bgr: np.ndarray) -> None:
        cv2.imshow(self.window_name, img_bgr)

    def snapshot(self, img_bgr: np.ndarray) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"snapshot_{_ts()}.jpg"
        cv2.imwrite(str(path), img_bgr)
        return path

    def _ensure_writer(self, frame_bgr: np.ndarray, fps: float) -> None:
        if self._writer is not None:
            return
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._record_path = self.runs_dir / f"record_{_ts()}.mp4"
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._record_path), fourcc, max(1.0, fps), (w, h))

    def set_recording(self, recording: bool, frame_bgr: np.ndarray, fps: float) -> None:
        if recording:
            self._ensure_writer(frame_bgr, fps)
        else:
            if self._writer is not None:
                self._writer.release()
            self._writer = None
            self._record_path = None

    def write_record_frame(self, frame_bgr: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        cv2.destroyWindow(self.window_name)


def _put_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    # deterministic vivid palette
    np.random.seed(track_id * 9973)
    c = np.random.randint(64, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def _ts() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


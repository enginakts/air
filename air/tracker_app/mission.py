from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .tracker import Track


@dataclass
class MissionEvent:
    kind: str  # "virtual_hit" | "friend_violation" | "stage_complete"
    track_id: int
    cls: int
    label: str
    dist_m: float


@dataclass
class MissionState:
    stage: int = 1  # 1..3
    hits: int = 0
    misses: int = 0
    friend_violations: int = 0

    # stage1 progress (0..2 for 3 hits)
    s1_step: int = 0


@dataclass(frozen=True)
class MotionInfo:
    vx_px_s: float
    vy_px_s: float
    dist_rate_dm_s: float  # negative means approaching (distance decreasing)


@dataclass(frozen=True)
class MissionConfig:
    # Stage 1: three steps at 5, 10, 15 meters in order (dm)
    s1_cls_step0: int = -1
    s1_cls_step1: int = -1
    s1_cls_step2: int = -1
    s1_dist_dm_step0: int = 50
    s1_dist_dm_step1: int = 100
    s1_dist_dm_step2: int = 150
    s1_tol_dm: int = 10  # +/- 1m default

    # Stage 2: moving target: trigger when within window around desired distance
    s2_desired_dm: int = 100
    s2_tol_dm: int = 10
    s2_approach_min_dm_s: float = 2.0  # require distance decreasing at least this fast to count "on time"

    # Stage 3: friend/enemy by balloon color; only enemy allowed
    s3_enemy_required: int = 1

    # General
    cooldown_s: float = 0.8


class MissionManager:
    def __init__(self) -> None:
        self.state = MissionState()
        self._last_hit_ts_by_id: Dict[int, float] = {}

    def set_stage(self, stage: int) -> None:
        stage = max(1, min(3, int(stage)))
        if stage != self.state.stage:
            self.state = MissionState(stage=stage)
            self._last_hit_ts_by_id.clear()

    def _cooldown_ok(self, track_id: int, now: float, cooldown_s: float) -> bool:
        prev = self._last_hit_ts_by_id.get(track_id, -1e9)
        if now - prev < cooldown_s:
            return False
        self._last_hit_ts_by_id[track_id] = now
        return True

    def step(
        self,
        *,
        now: float,
        engage_enabled: bool,
        crosshair_xy: Tuple[int, int],
        tracks: Sequence[Track],
        iff_by_id: Dict[int, Tuple[str, float]],
        dist_by_id: Dict[int, float],
        motion_by_id: Dict[int, MotionInfo],
        cfg: MissionConfig,
    ) -> Optional[MissionEvent]:
        if not engage_enabled:
            return None

        cx, cy = crosshair_xy

        # Determine which track is "selected" for interaction: prefer enemy under crosshair.
        selected: Optional[Track] = None
        for tr in tracks:
            lab, _ = iff_by_id.get(tr.track_id, ("unknown", 0.0))
            if lab != "enemy":
                continue
            if tr.x1 <= cx <= tr.x2 and tr.y1 <= cy <= tr.y2:
                selected = tr
                break

        # Friend violation if crosshair is inside friend box while engage is on.
        for tr in tracks:
            lab, _ = iff_by_id.get(tr.track_id, ("unknown", 0.0))
            if lab != "friend":
                continue
            if tr.x1 <= cx <= tr.x2 and tr.y1 <= cy <= tr.y2:
                if self._cooldown_ok(tr.track_id, now, cfg.cooldown_s):
                    self.state.friend_violations += 1
                    return MissionEvent(
                        kind="friend_violation",
                        track_id=tr.track_id,
                        cls=tr.cls,
                        label="friend",
                        dist_m=float(dist_by_id.get(tr.track_id, 0.0)),
                    )

        if selected is None:
            return None

        if not self._cooldown_ok(selected.track_id, now, cfg.cooldown_s):
            return None

        dist_m = float(dist_by_id.get(selected.track_id, 0.0))
        dist_dm = int(round(dist_m * 10.0)) if dist_m > 0 else 0
        motion = motion_by_id.get(selected.track_id)

        if self.state.stage == 1:
            return self._stage1_hit(now, selected, dist_dm, dist_m, cfg)
        if self.state.stage == 2:
            return self._stage2_hit(now, selected, dist_dm, dist_m, motion, cfg)
        return self._stage3_hit(now, selected, dist_m, cfg)

    def _stage1_hit(
        self,
        now: float,
        tr: Track,
        dist_dm: int,
        dist_m: float,
        cfg: MissionConfig,
    ) -> Optional[MissionEvent]:
        steps = [
            (cfg.s1_cls_step0, cfg.s1_dist_dm_step0),
            (cfg.s1_cls_step1, cfg.s1_dist_dm_step1),
            (cfg.s1_cls_step2, cfg.s1_dist_dm_step2),
        ]
        step = self.state.s1_step
        if step >= len(steps):
            return None

        want_cls, want_dm = steps[step]
        tol = max(0, int(cfg.s1_tol_dm))

        # If user didn't fill (want_cls < 0) treat as "any class".
        cls_ok = True if want_cls < 0 else (tr.cls == want_cls)
        dist_ok = (dist_dm > 0) and (want_dm - tol <= dist_dm <= want_dm + tol)

        if cls_ok and dist_ok:
            self.state.hits += 1
            self.state.s1_step += 1
            if self.state.s1_step >= 3:
                return MissionEvent(kind="stage_complete", track_id=tr.track_id, cls=tr.cls, label="stage1", dist_m=dist_m)
            return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label=f"s1_step{step}", dist_m=dist_m)

        self.state.misses += 1
        return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="miss_s1", dist_m=dist_m)

    def _stage2_hit(
        self,
        now: float,
        tr: Track,
        dist_dm: int,
        dist_m: float,
        motion: Optional[MotionInfo],
        cfg: MissionConfig,
    ) -> Optional[MissionEvent]:
        want = int(cfg.s2_desired_dm)
        tol = max(0, int(cfg.s2_tol_dm))

        # If user did not set desired distance (0), fall back to "fastest": accept any valid estimated distance.
        if want <= 0 or dist_dm <= 0:
            self.state.hits += 1
            return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="s2_fast", dist_m=dist_m)

        in_window = want - tol <= dist_dm <= want + tol
        approaching = False
        if motion is not None:
            # approaching if distance is decreasing fast enough
            approaching = motion.dist_rate_dm_s <= -abs(cfg.s2_approach_min_dm_s)

        if in_window and approaching:
            self.state.hits += 1
            return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="s2_on_time", dist_m=dist_m)

        self.state.misses += 1
        if in_window and not approaching:
            return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="miss_s2_not_approaching", dist_m=dist_m)
        return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="miss_s2", dist_m=dist_m)

    def _stage3_hit(self, now: float, tr: Track, dist_m: float, cfg: MissionConfig) -> Optional[MissionEvent]:
        # Stage 3: only enemy targets should be hit; selection already prefers enemy under crosshair.
        self.state.hits += 1
        return MissionEvent(kind="virtual_hit", track_id=tr.track_id, cls=tr.cls, label="s3_enemy", dist_m=dist_m)


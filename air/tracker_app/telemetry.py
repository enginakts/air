from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass
from typing import Optional, Protocol, Tuple


@dataclass(frozen=True)
class TelemetryPacket:
    ts: float
    stage: int
    track_id: int
    cls: int
    iff: str  # "friend" | "enemy" | "unknown"
    conf: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    center: Tuple[float, float]  # cx,cy
    vel_px_s: Tuple[float, float]  # vx,vy
    dist_m: float
    event: str  # "" or "virtual_hit" / "friend_violation" / "stage_complete"
    event_label: str


class TelemetrySink(Protocol):
    def send(self, pkt: TelemetryPacket) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def send(self, pkt: TelemetryPacket) -> None:
        return

    def close(self) -> None:
        return


class UdpTelemetrySink:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pkt: TelemetryPacket) -> None:
        payload = json.dumps(asdict(pkt), separators=(",", ":")).encode("utf-8")
        self.sock.sendto(payload, self.addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class SerialTelemetrySink:
    """
    Compact ASCII telemetry for STM32 over UART-USB.

    Line format:
      TEL,ts_ms,stage,track_id,cls,iff,conf_x100,bx1,by1,bx2,by2,cx,cy,vx,vy,dist_cm,event,event_label\n

    Notes:
    - cx,cy are integer pixels
    - vx,vy are px/s rounded to int
    - dist_cm is 0 if unknown
    """

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.0) -> None:
        try:
            import serial  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyserial not installed. Run: pip install -r requirements.txt") from e

        self.ser = serial.Serial(port=port, baudrate=int(baud), timeout=timeout_s, write_timeout=timeout_s)

    def send(self, pkt: TelemetryPacket) -> None:
        ts_ms = int(pkt.ts * 1000)
        conf_x100 = int(round(pkt.conf * 100))
        x1, y1, x2, y2 = pkt.bbox
        cx, cy = int(round(pkt.center[0])), int(round(pkt.center[1]))
        vx, vy = int(round(pkt.vel_px_s[0])), int(round(pkt.vel_px_s[1]))
        dist_cm = int(round(pkt.dist_m * 100.0)) if pkt.dist_m > 0 else 0
        # sanitize commas in labels
        ev = (pkt.event or "").replace(",", "_")
        lab = (pkt.event_label or "").replace(",", "_")
        line = f"TEL,{ts_ms},{pkt.stage},{pkt.track_id},{pkt.cls},{pkt.iff},{conf_x100},{x1},{y1},{x2},{y2},{cx},{cy},{vx},{vy},{dist_cm},{ev},{lab}\n"
        self.ser.write(line.encode("ascii", errors="ignore"))

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


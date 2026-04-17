from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Optional, Protocol, Tuple


@dataclass(frozen=True)
class FireCommand:
    """
    dx,dy: aim error normalized to [-1,1] where (0,0) is centered crosshair.
    fire: 1 only for the instant fire event (edge), not held high.
    """

    ts: float
    track_id: int
    cls: int
    iff: str  # "friend" | "enemy" | "unknown"
    dist_m: float
    dx: float
    dy: float
    fire: int


class OutputSink(Protocol):
    def send(self, cmd: FireCommand) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def send(self, cmd: FireCommand) -> None:
        return

    def close(self) -> None:
        return


class UdpSink:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: FireCommand) -> None:
        payload = json.dumps(asdict(cmd), separators=(",", ":")).encode("utf-8")
        self.sock.sendto(payload, self.addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class SerialSink:
    """
    Sends compact ASCII lines to STM32 over UART-USB.

    Default wire format (single line):
      CMD,ts_ms,track_id,cls,iff,dist_cm,dx_milli,dy_milli,fire\n
    """

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.0) -> None:
        try:
            import serial  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyserial not installed. Run: pip install -r requirements.txt") from e

        self._serial_mod = serial
        self.ser = serial.Serial(port=port, baudrate=int(baud), timeout=timeout_s, write_timeout=timeout_s)

    def send(self, cmd: FireCommand) -> None:
        ts_ms = int(cmd.ts * 1000)
        dist_cm = int(round(cmd.dist_m * 100.0)) if cmd.dist_m > 0 else 0
        dx_m = int(round(cmd.dx * 1000.0))
        dy_m = int(round(cmd.dy * 1000.0))
        line = f"CMD,{ts_ms},{cmd.track_id},{cmd.cls},{cmd.iff},{dist_cm},{dx_m},{dy_m},{cmd.fire}\n"
        self.ser.write(line.encode("ascii", errors="ignore"))

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


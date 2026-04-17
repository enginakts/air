"""
Run this on Jetson to receive aim/fire commands over UDP.

Example:
  python3 -m tracker_app.jetson_udp_receiver --bind 0.0.0.0 --port 5005

This file intentionally only prints the received command. You can hook it into:
- Jetson.GPIO PWM
- I2C PWM driver (PCA9685)
- CAN / serial link to motor controller
"""

from __future__ import annotations

import argparse
import json
import socket


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5005)
    return p.parse_args()


def main() -> int:
    a = _args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.bind, a.port))
    print(f"[udp_receiver] listening on {a.bind}:{a.port}")
    while True:
        data, addr = sock.recvfrom(65535)
        try:
            msg = json.loads(data.decode("utf-8", errors="replace"))
        except Exception:
            msg = {"raw": data[:200].decode("utf-8", errors="replace")}
        print(f"[udp_receiver] from={addr} msg={msg}")


if __name__ == "__main__":
    raise SystemExit(main())


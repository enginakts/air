"""
Ultralytics resmi YOLO12 ağırlıklarını `object_tracking/` klasörüne indirir.

Resmi isimler: yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
(https://docs.ultralytics.com/models/yolo12/)

İsteğe bağlı: nano (n) indirildikten sonra `yol12v.pt` takma adı (eski proje adı).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO12 COCO ağırlıklarını bu klasöre indir")
    p.add_argument(
        "--size",
        default="n",
        choices=("n", "s", "m", "l", "x"),
        help="Varyant: n=hızlı/küçük … x=en ağır (varsayılan: n)",
    )
    p.add_argument(
        "--alias-yol12v",
        action="store_true",
        help="yolo12n.pt indirildikten sonra aynı dosyayı yol12v.pt olarak da kopyala (UI varsayılan adı)",
    )
    args = p.parse_args()

    pkg = os.path.dirname(os.path.abspath(__file__))
    os.chdir(pkg)

    hub_name = f"yolo12{args.size}.pt"
    dest = os.path.join(pkg, hub_name)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Hata: ultralytics yüklü değil. Önce: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"[yolo12] indiriliyor / doğrulanıyor: {hub_name} (klasör: {pkg})")
    YOLO(hub_name)

    if not os.path.isfile(dest):
        # Bazı sürümlerde indirme cwd dışına düşebilir; en azından hub çağrısı tamamlandı
        cand = os.path.join(os.getcwd(), hub_name)
        if os.path.isfile(cand) and os.path.abspath(cand) != dest:
            shutil.copy2(cand, dest)
            print(f"[yolo12] kopyalandı: {cand} -> {dest}")

    if not os.path.isfile(dest):
        print(f"Uyarı: {dest} bulunamadı; Ultralytics önbelleğini kontrol edin.", file=sys.stderr)
        return 2

    print(f"[yolo12] hazır: {dest} ({os.path.getsize(dest) // (1024 * 1024)} MiB civarı)")

    if args.alias_yol12v and args.size == "n":
        alias = os.path.join(pkg, "yol12v.pt")
        shutil.copy2(dest, alias)
        print(f"[yolo12] takma ad: {alias}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

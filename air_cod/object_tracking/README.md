# Nesne Tespit + Takip (YOLO + DeepSORT + Kalman + OpenCV)

Bu mini proje:
- **YOLO (Ultralytics)** ile nesne **tespiti**
- **DeepSORT** ile **takip** (içinde **Kalman filtresi** kullanır)
- **OpenCV** ile kamera/video okuma ve çizim

## Kurulum

```bash
cd air_cod\object_tracking
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

### Webcam
```bash
python run_tracker.py --source 0
```

### UI (resimdeki arayüze benzer)
```bash
python ui_app.py
```

### Donanıma göre otomatik ayar (önerilen)
```bash
python run_tracker.py --source 0 --autotune --show-fps
```

### Video dosyası
```bash
python run_tracker.py --source "C:\path\to\video.mp4"
```

## Model (YOLO12)

Resmi ağırlıklar: `yolo12n.pt` … `yolo12x.pt` ([YOLO12 dokümantasyonu](https://docs.ultralytics.com/models/yolo12/)).

Bu klasöre indirmek için:

```bash
python download_yolo12_weights.py --size n
```

Eski proje adıyla uyum için (nano → `yol12v.pt` kopyası):

```bash
python download_yolo12_weights.py --size n --alias-yol12v
```

Ağırlıklar `air_cod/object_tracking/` içinde aranır; `run_tracker` varsayılanı `yolo12n.pt` (yoksa Ultralytics indirir).

```bash
python run_tracker.py --model yolo12s.pt
```


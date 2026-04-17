# Çelikkube – Nesne Tespit ve Takip (YOLO + DeepSORT + OpenCV)

Bu proje, **YOLO tabanlı tespit** ve **DeepSORT (Kalman filtresi) tabanlı takip** ile,
Çelikkube Hava Savunma Sistemleri yarışması için basit bir **OpenCV arayüzü** sağlar.

## Hedef sınıflar

Modelinizin şu 4 sınıfı öğrenmiş olması beklenir:

- `balistik_fuze`
- `helikopter`
- `savas_ucagi`
- `mini_micro_iha`

> Not: Elinizdeki çizim/3B hedeflere göre **özel eğitilmiş** bir YOLO ağırlığı (`.pt`) gerekli.

## Kurulum

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Çalıştırma

### Webcam
```bash
python -m tracker_app.main --weights weights/yolov12_celikkube.pt --source 0
```

### Video dosyası
```bash
python -m tracker_app.main --weights weights/yolov12_celikkube.pt --source data/sample.mp4
```

### Kamera seçtir (interaktif)
```bash
python -m tracker_app.main --weights weights/yolov12_celikkube.pt --select-camera
```

### STM32 + Jetson telemetri (simülasyon güvenli)
```bash
python -m tracker_app.main --weights weights/yolov12_celikkube.pt --select-camera --stm32-port COM3 --stm32-baud 115200 --jetson-udp 192.168.1.50:5005
```

## Simülasyon (Görev Aşamaları)

Bu sürümde sistem “ateş” yerine **virtual hit** olayları üretir ve ekranda skor/progress gösterir.

- **Stage 1**: 5m → 10m → 15m mesafede **belirtilen sırayla** (S1 step0/1/2 class trackbar’ları)
- **Stage 2**: hareketli hedef, `S2 want dm` ± `S2 tol dm` penceresinde “on time”
- **Stage 3**: 2 dost + 1 düşman; balon renginden IFF ile **sadece düşman** vurulur, dost üstünde Engage olursa ceza

## Kontroller (OpenCV penceresi)

- `q`: çıkış
- `space`: durdur/devam
- `r`: video kaydı aç/kapat (çıktı `runs/` altına)
- `s`: anlık görüntü kaydet (çıktı `runs/` altına)
- `1..4`: sınıf filtreleme aç/kapat (sol üstte durum yazar)
- `f`: tıklama ile **dost rengi** seçme modu
- `e`: tıklama ile **düşman rengi** seçme modu
- `c`: seçilen dost/düşman renklerini temizle
- `t`: **ateş izni** aç/kapat (yalnız düşman hedeflerde)
- `g`: **mesafe kapısı (RangeGate)** aç/kapat (sınıf aralığı doluysa uygular)
- `p`: referans mesafe için **bbox kalibrasyonu** (crosshair altındaki hedefin bbox genişliğini kaydeder)

Penceredeki trackbar’lar:

- `min color%`: kutu içinde renk eşleşme oranı eşiği
- `FR dh/ds/dv`: dost renk toleransı (HSV)
- `EN dh/ds/dv`: düşman renk toleransı (HSV)
- `ref dist dm`: referans mesafe (desimetre) — örn. 100 = 10.0 m
- `R0..R3 min/max dm`: sınıf bazlı vurma aralığı (desimetre). 0 bırakılırsa o sınıf için mesafe kısıtı uygulanmaz.
- `stage 1-3`: görev aşaması seçimi
- `S1 step0/1/2 cls`: Stage 1 sıra hedef sınıfı (0..3)
- `S1 tol dm`: Stage 1 mesafe toleransı
- `S2 want dm`, `S2 tol dm`: Stage 2 hedef mesafe penceresi

## Notlar

- DeepSORT zaten **Kalman filtresi** içerir. Ek olarak, projede her track için merkez noktası üzerinde
  opsiyonel bir **ek Kalman yumuşatma** da uygulanır.
- `--weights` yolunu kendi YOLOv12/YOLO ağırlığınıza göre verin. (Ultralytics `YOLO(...)` loader’ı ile yüklenir.)
- “FIRE” yazısı simülasyonda “virtual hit” anını temsil eder.
- Mesafe tahmini **tek kamerada yaklaşık** yapılır: referans mesafede alınan bbox genişliği ile \(\text{mesafe} \propto 1/\text{bbox\_genişliği}\).
- Stage 2’de “tam zamanında” sayılması için hedefin mesafe penceresine girerken **yaklaşıyor olması** (mesafenin azalması) şartı uygulanır.


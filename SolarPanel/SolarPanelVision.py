import cv2
import torch
from ultralytics import YOLO

# Kamera ayarları (0, varsayılan olarak ilk kamerayı kullanır)
cap = cv2.VideoCapture("video.mp4")  # Kamera bağlantısını başlat

# CUDA kontrolü (GPU kullanılabilir mi?)
if torch.cuda.is_available():
    device = 'cuda'
    print(f"CUDA mevcut! GPU adı: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("CUDA mevcut değil, CPU kullanılacak.")

# YOLOv11 modelini yükle (best.pt, sizin modeliniz)
model = YOLO('best.pt')  # best.pt dosyanızın doğru dizinde olduğundan emin olun
model.to(device)  # CUDA'ya taşı (eğer mevcutsa)

# Sınıf etiketleri (etiketlerin sırasına dikkat edin)
class_names = ['bird drop', 'clean', 'dusty', 'electrical damage', 'physical damage', 'snow covered']

# Sonsuz döngü, kamera sürekli çalışacak
while True:
    ret, frame = cap.read()  # Kameradan bir kare al

    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    # YOLO tahmini (inference) yap
    results = model(frame)

    # Sonuçların çerçevelerini ve sınıf etiketlerini çizin
    for result in results:
        boxes = result.boxes  # Tüm algılanan kutuları al

        for box in boxes:
            # Koordinatlar ve skor
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Kutu koordinatları
            class_id = int(box.cls[0])  # Sınıf kimliği
            confidence = float(box.conf[0])  # Güven puanı

            # Sınıf adını ve güven puanını al
            label = f"{class_names[class_id]} ({confidence*100:.2f}%)"

            # Kutu (Bounding box) çizimi (yeşil renk, 2px kalınlık)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Metin (sınıf adı + güven puanı) ekleme
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonuçları ekranda göster
    cv2.imshow('YOLOv11 Algılama', frame)

    # ESC tuşuna basarak çıkma
    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu
        break

# Kamera bağlantısını ve tüm OpenCV pencerelerini kapat
cap.release()
cv2.destroyAllWindows()

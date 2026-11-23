from ultralytics import YOLO

if __name__ == '__main__':
    # Muat bobot dasar YOLOv8s
    model = YOLO('yolov8s.pt')

    # Jalankan pelatihan dengan dataset v7i
    results = model.train(
        data='Ketupat.v7i.yolov8/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        name='yolov8_ketupat_v7i',
        # Pertahankan konfigurasi augmentasi optimal
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
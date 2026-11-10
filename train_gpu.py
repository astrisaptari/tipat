from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    model.train(
        # 1. Ubah path data ke folder v6i yang baru
        data='Ketupat.v6i.yolov8/data.yaml', 
        
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        
        # Parameter Augmentasi YOLOv8
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # 2. Ubah nama folder output agar hasil pelatihan tersimpan rapi
        name='yolov8_ketupat_v6i'
    )
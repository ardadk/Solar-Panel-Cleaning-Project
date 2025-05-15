from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="video.mp4", show=True)
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

metrics = model.val(
    data="data.yaml",
    split="test"
)

print(metrics)
print()

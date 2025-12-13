# train.py
from ultralytics import YOLO
import torch

# Проверка GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

# Загрузка предобученной модели (yolov8s — хороший баланс скорости/точности)
model = YOLO('yolov8s.pt')  # или yolov8n.pt для быстрее, yolov8m.pt для лучше

# Обучение
results = model.train(
    data='data/data.yaml',   # путь к вашему data.yaml
    epochs=100,              # можно 50-100
    imgsz=640,
    batch=16,                # подберите под вашу видеокарту (8-32)
    name='cat_breeds_yolov8',
    patience=20,             # early stopping
    device=device,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001
)

# Сохранение лучшей модели
model.export(format='onnx')  # опционально экспорт в ONNX



from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=400,
        batch=8,
        optimizer="Adam",
        lr0=0.001,

        classes=[0, 1, 2, 3, 4, 5],

        device="cpu"  
    )

if __name__ == "__main__":
    main()

from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "durian-disease-split" / "data.yaml"

MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

EPOCHS = 80
IMAGE_SIZE = 640
PATIENCE = 20

PROJECT_NAME = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "durian_yolov8"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {DATA_YAML}")

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = 0
        batch_size = 16
        workers = 4
    else:
        device = "cpu"
        batch_size = 4
        workers = 0

    print("=" * 60)
    print("TRAIN YOLOv8")
    print("=" * 60)
    print("Data yaml:", DATA_YAML)
    print("Model:", MODEL_PATH)
    print("CUDA available:", cuda_available)
    print("Device:", device)
    print("Batch size:", batch_size)

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch_size,
        device=device,
        workers=workers,
        patience=PATIENCE,
        project=str(PROJECT_NAME),
        name=RUN_NAME,
        exist_ok=True,
        plots=True
    )

    print("Train YOLO xong.")
    print("Model tốt nhất nằm ở:")
    print(PROJECT_NAME / RUN_NAME / "weights" / "best.pt")


if __name__ == "__main__":
    main()

from pathlib import Path
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFICATION_DIR = PROJECT_ROOT / "durian-classification"
MODEL_PATH = PROJECT_ROOT / "best_durian_classifier.pth"


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

    checkpoint = load_checkpoint(MODEL_PATH, map_location="cpu")

    model_name = checkpoint["model_name"]
    image_size = checkpoint["image_size"]
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    test_dir = CLASSIFICATION_DIR / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy test folder: {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv("confusion_matrix.csv", encoding="utf-8-sig")

    print("Confusion matrix:")
    print(cm_df)
    print("Đã lưu confusion_matrix.csv")


if __name__ == "__main__":
    main()

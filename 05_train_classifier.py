from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFICATION_DIR = PROJECT_ROOT / "durian-classification"

MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

OUTPUT_MODEL = PROJECT_ROOT / "best_durian_classifier.pth"
OUTPUT_CLASSES = PROJECT_ROOT / "class_names.json"


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, valid_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


def main():
    train_dir = CLASSIFICATION_DIR / "train"
    valid_dir = CLASSIFICATION_DIR / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy train folder: {train_dir}")

    if not valid_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy valid folder: {valid_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        batch_size = 32
        num_workers = 4
    else:
        batch_size = 8
        num_workers = 0

    train_transform, valid_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=valid_dir,
        transform=valid_transform
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("=" * 60)
    print("TRAIN CLASSIFIER")
    print("=" * 60)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    print("Train images:", len(train_dataset))
    print("Valid images:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_valid_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_acc = evaluate(
            model,
            valid_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Valid Loss: {valid_loss:.4f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

            checkpoint = {
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "class_names": class_names,
                "num_classes": num_classes,
                "best_valid_acc": best_valid_acc,
                "model_state_dict": model.state_dict()
            }

            torch.save(checkpoint, OUTPUT_MODEL)

            with open(OUTPUT_CLASSES, "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)

            print("Đã lưu model tốt nhất:", OUTPUT_MODEL)

    print("Train classifier xong.")
    print("Best valid accuracy:", best_valid_acc)


if __name__ == "__main__":
    main()

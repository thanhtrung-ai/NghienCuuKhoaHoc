import random
import shutil
from pathlib import Path
import yaml

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATASET_DIR = PROJECT_ROOT / "durian-disease.v1-amag.yolov8"
OUTPUT_DATASET_DIR = PROJECT_ROOT / "durian-disease-split"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    image_dir = SOURCE_DATASET_DIR / "train" / "images"
    label_dir = SOURCE_DATASET_DIR / "train" / "labels"
    data_yaml = SOURCE_DATASET_DIR / "data.yaml"

    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy images: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy labels: {label_dir}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_yaml}")

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    split_data = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }

    for split_name, split_images in split_data.items():
        output_image_dir = OUTPUT_DATASET_DIR / split_name / "images"
        output_label_dir = OUTPUT_DATASET_DIR / split_name / "labels"

        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            label_path = label_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, output_image_dir / image_path.name)

            if label_path.exists():
                shutil.copy2(label_path, output_label_dir / label_path.name)
            else:
                print("Thiếu label:", label_path)

    config = load_yaml(data_yaml)

    config["path"] = str(OUTPUT_DATASET_DIR).replace("\\", "/")
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    save_yaml(OUTPUT_DATASET_DIR / "data.yaml", config)

    print("Đã chia dataset xong.")
    print("Tổng ảnh:", total)
    print("Train:", len(split_data["train"]))
    print("Valid:", len(split_data["valid"]))
    print("Test:", len(split_data["test"]))
    print("Dataset mới:", OUTPUT_DATASET_DIR)


if __name__ == "__main__":
    main()

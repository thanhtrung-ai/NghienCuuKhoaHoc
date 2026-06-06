from collections import Counter
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "durian-disease.v1-amag.yolov8"
DATA_YAML = DATASET_DIR / "data.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_KEYS = ["train", "val", "test"]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("names trong data.yaml phai la list hoac dict.")


def resolve_image_dir(split_name, config):
    raw_path = config.get(split_name)
    candidates = []

    if raw_path:
        raw_path = Path(str(raw_path))
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(DATASET_DIR / raw_path)
            candidates.append(DATA_YAML.parent / raw_path)

    candidates.append(DATASET_DIR / split_name / "images")

    if split_name == "val":
        candidates.append(DATASET_DIR / "valid" / "images")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def label_dir_from_image_dir(image_dir):
    return image_dir.parent / "labels"


def classify_label_line(parts):
    if len(parts) == 5:
        return "bbox"
    if len(parts) >= 7 and len(parts) % 2 == 1:
        return "segmentation"
    return None


def check_label_values(parts, num_classes):
    label_type = classify_label_line(parts)
    if label_type is None:
        return None, "Sai so luong gia tri: bbox can 5, segmentation can 1 + cac cap toa do"

    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return label_type, "Co gia tri khong phai so"

    if class_id < 0 or class_id >= num_classes:
        return label_type, "class_id vuot so class"

    if not all(0 <= value <= 1 for value in values):
        return label_type, "toa do khong nam trong khoang 0 den 1"

    return label_type, None


def check_split(split_name, config, num_classes):
    image_dir = resolve_image_dir(split_name, config)
    label_dir = label_dir_from_image_dir(image_dir)

    if not image_dir.exists():
        print(f"[SKIP] Khong co thu muc: {image_dir}")
        return

    if not label_dir.exists():
        print(f"[ERROR] Co images nhung khong co labels: {label_dir}")
        return

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    missing_labels = []
    bad_labels = []
    class_counter = Counter()
    label_type_counter = Counter()

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line_index, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            label_type, error = check_label_values(parts, num_classes)

            if error:
                bad_labels.append((label_path.name, line_index, line, error))
                continue

            class_id = int(float(parts[0]))
            class_counter[class_id] += 1
            label_type_counter[label_type] += 1

    print("=" * 60)
    print(f"CHECK SPLIT: {split_name}")
    print("=" * 60)
    print("Thu muc anh:", image_dir)
    print("Thu muc nhan:", label_dir)
    print("So anh:", len(images))
    print("So anh thieu label:", len(missing_labels))
    print("So dong label loi:", len(bad_labels))

    print("\nLoai label:")
    print("bbox:", label_type_counter["bbox"])
    print("segmentation:", label_type_counter["segmentation"])

    print("\nPhan bo class:")
    for class_id in range(num_classes):
        print(f"class {class_id}: {class_counter[class_id]} object")

    if missing_labels[:10]:
        print("\nVi du anh thieu label:")
        for item in missing_labels[:10]:
            print(item)

    if bad_labels[:10]:
        print("\nVi du label loi:")
        for item in bad_labels[:10]:
            print(item)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {DATASET_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Khong tim thay data.yaml: {DATA_YAML}")

    config = load_yaml(DATA_YAML)
    names = normalize_names(config["names"])
    num_classes = len(names)

    print("=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print("Dataset path:", DATASET_DIR)

    print("\nClasses:")
    for class_id, class_name in names.items():
        print(f"{class_id}: {class_name}")

    checked_dirs = set()
    for split_name in SPLIT_KEYS:
        if split_name not in config:
            continue

        image_dir = resolve_image_dir(split_name, config)
        image_dir_key = str(image_dir).lower()
        if image_dir_key in checked_dirs:
            continue

        checked_dirs.add(image_dir_key)
        check_split(split_name, config, num_classes)


if __name__ == "__main__":
    main()

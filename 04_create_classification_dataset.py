import cv2
from pathlib import Path
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_DATASET_DIR = PROJECT_ROOT / "durian-disease-split"
CLASSIFICATION_OUTPUT_DIR = PROJECT_ROOT / "durian-classification"

PADDING_RATIO = 0.15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    raise ValueError("names trong data.yaml không hợp lệ.")


def convert_yolo_to_xyxy(line, image_width, image_height):
    parts = line.strip().split()

    if len(parts) != 5:
        return None

    class_id = int(float(parts[0]))

    x_center = float(parts[1]) * image_width
    y_center = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    return class_id, x1, y1, x2, y2


def add_padding(x1, y1, x2, y2, image_width, image_height):
    padding_x = int((x2 - x1) * PADDING_RATIO)
    padding_y = int((y2 - y1) * PADDING_RATIO)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(image_width, x2 + padding_x)
    y2 = min(image_height, y2 + padding_y)

    return x1, y1, x2, y2


def process_split(split_name, class_names):
    image_dir = YOLO_DATASET_DIR / split_name / "images"
    label_dir = YOLO_DATASET_DIR / split_name / "labels"

    if not image_dir.exists():
        print(f"Bỏ qua vì không có: {image_dir}")
        return

    if not label_dir.exists():
        print(f"Bỏ qua vì không có: {label_dir}")
        return

    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    saved_count = 0

    for image_path in tqdm(image_paths, desc=f"Processing {split_name}"):
        image = cv2.imread(str(image_path))

        if image is None:
            print("Không đọc được ảnh:", image_path)
            continue

        image_height, image_width = image.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for box_index, line in enumerate(lines):
            parsed = convert_yolo_to_xyxy(line, image_width, image_height)

            if parsed is None:
                continue

            class_id, x1, y1, x2, y2 = parsed

            if class_id not in class_names:
                continue

            x1, y1, x2, y2 = add_padding(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            class_name = class_names[class_id].replace(" ", "_")

            save_dir = CLASSIFICATION_OUTPUT_DIR / split_name / class_name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"{image_path.stem}_{box_index}.jpg"
            cv2.imwrite(str(save_path), crop)

            saved_count += 1

    print(f"{split_name}: đã lưu {saved_count} ảnh crop.")


def main():
    data_yaml = YOLO_DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_yaml}")

    class_names = load_class_names(data_yaml)

    print("Class names:")
    for class_id, class_name in class_names.items():
        print(class_id, class_name)

    for split_name in ["train", "valid", "test"]:
        process_split(split_name, class_names)

    print("Tạo classification dataset xong.")
    print("Lưu tại:", CLASSIFICATION_OUTPUT_DIR)


if __name__ == "__main__":
    main()

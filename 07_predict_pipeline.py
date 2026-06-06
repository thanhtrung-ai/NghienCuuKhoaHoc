from pathlib import Path
import argparse
import json

import cv2
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "durian_yolov8" / "weights" / "best.pt"
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "best_durian_classifier.pth"

DEFAULT_INPUT_PATH = PROJECT_ROOT / "sample_predict_images"
DEFAULT_OUTPUT_IMAGE_PATH = PROJECT_ROOT / "final_output.jpg"
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "sample_predict_outputs"

YOLO_CONF_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict durian disease from one image or a folder of images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to one image or one folder containing images."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path if input is one image, or output folder if input is a folder."
    )
    return parser.parse_args()

def load_models(device):
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find YOLO model: {YOLO_MODEL_PATH}")

    if not CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find classifier model: {CLASSIFIER_MODEL_PATH}")

    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    checkpoint = load_checkpoint(CLASSIFIER_MODEL_PATH, map_location=device)
    model_name = checkpoint["model_name"]
    image_size = checkpoint["image_size"]
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    classifier = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return yolo_model, classifier, transform, class_names

def predict_one_image(
    image_path,
    output_path,
    yolo_model,
    classifier,
    transform,
    class_names,
    device,
    yolo_device
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    results = yolo_model(
        str(image_path),
        device=yolo_device,
        conf=YOLO_CONF_THRESHOLD
    )

    predictions = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detection_confidence = float(box.conf[0])

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            input_tensor = transform(crop_pil)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = classifier(input_tensor)
                probabilities = F.softmax(output, dim=1)

                class_id = torch.argmax(probabilities, dim=1).item()
                classification_confidence = float(probabilities[0][class_id])

            disease_name = class_names[class_id]
            final_confidence = detection_confidence * classification_confidence

            predictions.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "detection_confidence": detection_confidence,
                "classification_confidence": classification_confidence,
                "final_confidence": final_confidence,
                "disease": disease_name
            })

            label = f"{disease_name}: {final_confidence:.2f}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    return {
        "image_path": str(image_path),
        "output_image_path": str(output_path),
        "number_of_detections": len(predictions),
        "predictions": predictions
    }

def get_image_paths(input_folder):
    return sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def main():
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if torch.cuda.is_available() else "cpu"

    yolo_model, classifier, transform, class_names = load_models(device)

    if input_path.is_file():
        output_path = args.output or DEFAULT_OUTPUT_IMAGE_PATH
        result = predict_one_image(
            input_path,
            output_path,
            yolo_model,
            classifier,
            transform,
            class_names,
            device,
            yolo_device
        )

        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        for item in result["predictions"]:
            print(item)
        print("Saved output image:", output_path)
        return

    output_folder = args.output or DEFAULT_OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No image files found in folder: {input_path}")

    all_results = []

    for image_path in image_paths:
        output_image_path = output_folder / image_path.name
        result = predict_one_image(
            image_path,
            output_image_path,
            yolo_model,
            classifier,
            transform,
            class_names,
            device,
            yolo_device
        )
        all_results.append(result)

    results_path = output_folder / "prediction_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Number of images:", len(all_results))
    print("Saved output folder:", output_folder)
    print("Saved prediction file:", results_path)

if __name__ == "__main__":
    main()

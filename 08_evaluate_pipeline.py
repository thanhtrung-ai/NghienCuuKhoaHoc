from pathlib import Path
import argparse
import csv
import json

import cv2
import torch
import timm
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from ultralytics import YOLO
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "durian-disease-split"
DATA_YAML = DATASET_DIR / "data.yaml"
YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "durian_yolov8" / "weights" / "best.pt"
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "best_durian_classifier.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline_eval_outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO + classifier pipeline on a YOLO-format split."
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def load_class_names(data_yaml):
    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("names trong data.yaml phai la list hoac dict.")


def load_models(device):
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find YOLO model: {YOLO_MODEL_PATH}")
    if not CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find classifier model: {CLASSIFIER_MODEL_PATH}")

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    checkpoint = load_checkpoint(CLASSIFIER_MODEL_PATH, map_location=device)

    classifier = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()

    image_size = checkpoint["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return yolo_model, classifier, transform, checkpoint["class_names"]


def yolo_box_to_xyxy(parts, image_width, image_height):
    if len(parts) != 5:
        return None

    class_id = int(float(parts[0]))
    x_center = float(parts[1]) * image_width
    y_center = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x1 = max(0, int(x_center - box_width / 2))
    y1 = max(0, int(y_center - box_height / 2))
    x2 = min(image_width, int(x_center + box_width / 2))
    y2 = min(image_height, int(y_center + box_height / 2))

    return {
        "class_id": class_id,
        "bbox": [x1, y1, x2, y2],
    }


def load_ground_truth(label_path, image_width, image_height):
    boxes = []
    skipped = 0

    if not label_path.exists():
        return boxes, skipped

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        parsed = yolo_box_to_xyxy(parts, image_width, image_height)
        if parsed is None:
            skipped += 1
            continue

        boxes.append(parsed)

    return boxes, skipped


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def classify_crop(crop, classifier, transform, class_names, device):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(input_tensor)
        probabilities = F.softmax(output, dim=1)
        class_id = torch.argmax(probabilities, dim=1).item()
        confidence = float(probabilities[0][class_id])

    return class_id, class_names[class_id], confidence


def predict_image(image_path, yolo_model, classifier, transform, class_names, device, yolo_device, conf):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    results = yolo_model(str(image_path), device=yolo_device, conf=conf, verbose=False)
    predictions = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, min(x1, image.shape[1]))
            x2 = max(0, min(x2, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            y2 = max(0, min(y2, image.shape[0]))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            class_id, disease_name, classification_confidence = classify_crop(
                crop,
                classifier,
                transform,
                class_names,
                device,
            )
            detection_confidence = float(box.conf[0])

            predictions.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": int(class_id),
                "disease": disease_name,
                "detection_confidence": detection_confidence,
                "classification_confidence": classification_confidence,
                "final_confidence": detection_confidence * classification_confidence,
            })

    return image, predictions


def match_predictions(ground_truths, predictions, iou_threshold):
    candidates = []
    for gt_index, gt in enumerate(ground_truths):
        for pred_index, pred in enumerate(predictions):
            iou = box_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue

        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        gt = ground_truths[gt_index]
        pred = predictions[pred_index]
        matches.append({
            "iou": iou,
            "gt_class_id": gt["class_id"],
            "pred_class_id": pred["class_id"],
            "correct_class": gt["class_id"] == pred["class_id"],
            "prediction": pred,
        })

    return matches, matched_gt, matched_pred


def draw_predictions(image, predictions, class_names):
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        label = f"{pred['disease']}: {pred['final_confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return image


def save_report(output_dir, summary, per_image_rows, y_true, y_pred, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "pipeline_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "pipeline_eval_details.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    if y_true and y_pred:
        report_text = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    else:
        report_text = "No matched predictions for classification report."
        cm = []

    markdown = [
        "# Pipeline End-to-End Evaluation",
        "",
        "## Summary",
        "",
        f"- Images: {summary['images']}",
        f"- Ground-truth bbox objects: {summary['ground_truth_objects']}",
        f"- Skipped non-bbox labels: {summary['skipped_non_bbox_labels']}",
        f"- Predictions: {summary['predictions']}",
        f"- Matched detections: {summary['matched_detections']}",
        f"- Missed ground-truth objects: {summary['missed_ground_truth_objects']}",
        f"- False-positive detections: {summary['false_positive_detections']}",
        f"- Detection recall @ IoU {summary['iou_threshold']}: {summary['detection_recall']:.4f}",
        f"- Detection precision @ IoU {summary['iou_threshold']}: {summary['detection_precision']:.4f}",
        f"- End-to-end class accuracy on matched detections: {summary['matched_class_accuracy']:.4f}",
        "",
        "## Classification Report On Matched Detections",
        "",
        "```text",
        report_text,
        "```",
        "",
        "## Confusion Matrix",
        "",
        "```text",
        str(cm),
        "```",
    ]

    (output_dir / "pipeline_eval_report.md").write_text("\n".join(markdown), encoding="utf-8")


def main():
    args = parse_args()
    class_id_to_name = load_class_names(DATA_YAML)
    class_names = [class_id_to_name[i] for i in sorted(class_id_to_name)]

    image_dir = DATASET_DIR / args.split / "images"
    label_dir = DATASET_DIR / args.split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find image folder: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Cannot find label folder: {label_dir}")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
    yolo_model, classifier, transform, classifier_class_names = load_models(device)

    if list(classifier_class_names) != class_names:
        raise ValueError(
            f"Classifier classes {classifier_class_names} do not match dataset classes {class_names}"
        )

    image_output_dir = args.output / "images"
    if args.save_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    total_gt = 0
    total_predictions = 0
    total_matches = 0
    total_class_correct = 0
    skipped_non_bbox = 0
    y_true = []
    y_pred = []
    per_image_rows = []

    for image_path in image_paths:
        image, predictions = predict_image(
            image_path,
            yolo_model,
            classifier,
            transform,
            class_names,
            device,
            yolo_device,
            args.conf,
        )
        image_height, image_width = image.shape[:2]
        ground_truths, skipped = load_ground_truth(
            label_dir / f"{image_path.stem}.txt",
            image_width,
            image_height,
        )
        matches, matched_gt, matched_pred = match_predictions(ground_truths, predictions, args.iou)

        class_correct = sum(1 for item in matches if item["correct_class"])
        total_gt += len(ground_truths)
        total_predictions += len(predictions)
        total_matches += len(matches)
        total_class_correct += class_correct
        skipped_non_bbox += skipped

        for item in matches:
            y_true.append(item["gt_class_id"])
            y_pred.append(item["pred_class_id"])

        per_image_rows.append({
            "image": image_path.name,
            "ground_truth_objects": len(ground_truths),
            "predictions": len(predictions),
            "matched_detections": len(matches),
            "class_correct_matches": class_correct,
            "missed_ground_truth_objects": len(ground_truths) - len(matched_gt),
            "false_positive_detections": len(predictions) - len(matched_pred),
            "skipped_non_bbox_labels": skipped,
        })

        if args.save_images:
            annotated = draw_predictions(image.copy(), predictions, class_names)
            cv2.imwrite(str(image_output_dir / image_path.name), annotated)

    detection_recall = total_matches / total_gt if total_gt else 0.0
    detection_precision = total_matches / total_predictions if total_predictions else 0.0
    matched_class_accuracy = total_class_correct / total_matches if total_matches else 0.0

    summary = {
        "split": args.split,
        "images": len(image_paths),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "ground_truth_objects": total_gt,
        "skipped_non_bbox_labels": skipped_non_bbox,
        "predictions": total_predictions,
        "matched_detections": total_matches,
        "missed_ground_truth_objects": total_gt - total_matches,
        "false_positive_detections": total_predictions - total_matches,
        "detection_recall": detection_recall,
        "detection_precision": detection_precision,
        "matched_class_accuracy": matched_class_accuracy,
    }

    save_report(args.output, summary, per_image_rows, y_true, y_pred, class_names)

    print("=" * 60)
    print("PIPELINE END-TO-END EVALUATION")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved report:", args.output / "pipeline_eval_report.md")


if __name__ == "__main__":
    main()

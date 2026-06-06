# Durian Disease Detection

Prototype nhận diện bệnh trên trái sầu riêng bằng pipeline hai giai đoạn:

1. YOLOv8 phát hiện vùng bệnh.
2. EfficientNet-B0 phân loại vùng crop thành `mold` hoặc `rot`.

## Cấu Trúc Chính

| File/thư mục | Vai trò |
|---|---|
| `durian-disease.v1-amag.yolov8/` | Dataset gốc định dạng YOLO |
| `sample_predict_images/` | Ảnh mẫu để chạy dự đoán nhanh |
| `01_check_dataset.py` | Kiểm tra ảnh, label và phân bố class |
| `02_split_dataset.py` | Chia dataset thành train/valid/test |
| `03_train_yolo.py` | Huấn luyện YOLOv8 |
| `04_create_classification_dataset.py` | Crop bbox YOLO thành dataset classification |
| `05_train_classifier.py` | Huấn luyện EfficientNet-B0 |
| `06_evaluate_classifier.py` | Đánh giá classifier trên tập crop test |
| `07_predict_pipeline.py` | Dự đoán một ảnh hoặc cả folder ảnh |
| `08_evaluate_pipeline.py` | Đánh giá pipeline YOLO + classifier end-to-end |
| `yolov8n.pt` | Weight YOLOv8n ban đầu để train |

Các thư mục/file sinh ra khi chạy như `durian-disease-split/`, `durian-classification/`, `runs/`, `sample_predict_outputs/`, `best_durian_classifier.pth`, `confusion_matrix.csv` được đưa vào `.gitignore` và không upload lên GitHub.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics torch torchvision timm opencv-python pillow scikit-learn pandas pyyaml tqdm
```

Nếu máy có GPU NVIDIA, nên cài PyTorch theo hướng dẫn CUDA phù hợp từ trang chính thức của PyTorch.

## Quy Trình Chạy Lại

Kiểm tra dataset gốc:

```powershell
python 01_check_dataset.py
```

Chia dataset thành train/valid/test:

```powershell
python 02_split_dataset.py
```

Huấn luyện YOLOv8:

```powershell
python 03_train_yolo.py
```

Tạo dataset crop cho classifier:

```powershell
python 04_create_classification_dataset.py
```

Huấn luyện classifier:

```powershell
python 05_train_classifier.py
```

Đánh giá classifier:

```powershell
python 06_evaluate_classifier.py
```

Dự đoán cả folder mặc định `sample_predict_images/`:

```powershell
python 07_predict_pipeline.py
```

Dự đoán folder khác:

```powershell
python 07_predict_pipeline.py --input path\to\images --output path\to\outputs
```

Dự đoán một ảnh:

```powershell
python 07_predict_pipeline.py --input path\to\image.jpg --output result.jpg
```

Đánh giá end-to-end trên tập test:

```powershell
python 08_evaluate_pipeline.py --split test --output pipeline_eval_outputs
```

Nếu muốn lưu ảnh có bounding box khi đánh giá:

```powershell
python 08_evaluate_pipeline.py --split test --output pipeline_eval_outputs --save-images
```

## Ghi Chú

- Dataset hiện có 2 class: `mold` và `rot`.
- `02_split_dataset.py` chia theo số ảnh, không chia theo số object.
- `04_create_classification_dataset.py` chỉ dùng label bbox 5 giá trị; label segmentation không được crop trực tiếp trong phiên bản hiện tại.
- Muốn thêm loại bệnh mới cần thêm class trong `data.yaml`, annotate object class mới, rồi train lại YOLO và classifier.

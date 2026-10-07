# 🗺️ Bản Đồ Code (Codebase Map)
> File nào làm gì, gọi file nào, quan trọng như thế nào.

---

## Sơ đồ phụ thuộc

```
11_webcam.py (ENTRY POINT - chạy webcam)
    └── inference.py
            ├── model_common.py (load model)
            │       └── preprocessing.py (hằng số padding)
            └── preprocessing.py (add_image_padding, normalize_landmarks)

10_predict_image.py (predict 1 ảnh tĩnh)
    └── inference.py (same as above)

run_from_07.py (chạy pipeline training từ bước 7)
    ├── 07_prepare_splits.py
    ├── 08_train_models.py
    └── 09_evaluate_model.py
```

---

## Chi tiết từng file

### 🔴 File Entry Points (files chạy trực tiếp)

#### `11_webcam.py` ⭐ FILE QUAN TRỌNG NHẤT
- **Mục đích**: Real-time webcam recognition
- **Chạy**: `python 11_webcam.py --camera 0`
- **Phụ thuộc**: `inference.py`
- **Luồng**: Mở webcam → mỗi frame gọi `recognizer.process(frame)` → vẽ kết quả → hiển thị

#### `10_predict_image.py`
- **Mục đích**: Predict 1 ảnh tĩnh từ file
- **Chạy**: `python 10_predict_image.py "path/to/image.jpg" --show`
- **Dùng cho**: Testing nhanh, debug

#### `run_from_07.py`
- **Mục đích**: Chạy toàn bộ pipeline training (từ step 07)
- **Chạy**: `python run_from_07.py`
- **Dùng cho**: Re-train model nếu muốn

### 🟡 File Shared Modules (thư viện dùng chung)

#### `preprocessing.py` ⭐ FILE CỐT LÕI
- **Hàm chính**:
  - `add_image_padding(image_rgb)` → thêm padding đen 25% mỗi cạnh
  - `normalize_landmarks(landmarks_xy, width, height)` → chuẩn hóa 42 features
- **Hằng số**: `IMAGE_PADDING_FRACTION = 0.25`
- **Quan trọng**: File này PHẢI giống nhau giữa training và inference (có hash check)
- **Demo**: Chạy `python preprocessing.py` để xem ví dụ tính toán

#### `inference.py` ⭐ FILE TRUNG TÂM
- **Class `SignRecognizer`**:
  - `__init__`: Load MediaPipe model + sklearn model
  - `process(image_bgr)`: Pipeline đầy đủ 1 frame → trả `FrameResult`
  - `close()`: Giải phóng MediaPipe detector
- **Class `PredictionSmoother`**: Làm mượt dự đoán qua nhiều frame
- **Hàm `draw_landmarks`**: Vẽ 21 điểm lên ảnh để hiển thị
- **Hàm `status_message`**: Chuyển status code thành text hiển thị

#### `model_common.py`
- **Mục đích**: Contract & utilities dùng chung giữa training và inference
- **Quan trọng nhất**:
  - `LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")` — 36 class
  - `FEATURE_COLUMNS` — tên 42 cột features
  - `PREPROCESSING` — dict mô tả preprocessing settings
  - `load_bundle()` — load model với validation hash
  - `load_split()` — load train/val/test CSV với validation

### 🟢 File Pipeline Scripts (chạy theo thứ tự để training)

| File | Bước | Mục đích | Input | Output |
|------|------|----------|-------|--------|
| `01_explore_data.py` | 1 | Khám phá dataset | data/raw/ | stats |
| `02_inspect_hand.py` | 2 | Xem MediaPipe trên 1 ảnh | 1 ảnh | viz |
| `03_extract_features.py` | 3 | Extract không padding (baseline) | data/raw/ | features.csv |
| `04_review_extraction.py` | 4 | Review kết quả extraction | outputs/ | report |
| `05_test_padding.py` | 5 | Thử nghiệm padding | data/raw/ | comparison |
| `06_extract_padded_features.py` | 6 | Extract có padding (final) | data/raw/ | features_padded.csv |
| `07_prepare_splits.py` | 7 | Chia train/val/test | features_padded.csv | splits/ |
| `08_train_models.py` | 8 | Train 3 model, chọn best | splits/ | sign_classifier.joblib |
| `09_evaluate_model.py` | 9 | Evaluate trên test set | model + test.csv | metrics |

> ⚠️ Steps 01-06 cần raw dataset (không có trong repo). Steps 07-09 có thể chạy vì repo đã có processed data.

### 📁 Thư mục

```
models/
├── hand_landmarker.task    # MediaPipe pre-trained model (7.8 MB) — Google cung cấp
└── sign_classifier.joblib  # Sklearn Logistic Regression model (14 KB)

data/processed/splits/
├── train.csv               # 2387 rows × 43 cols (42 features + label)
├── validation.csv          # 507 rows
├── test.csv                # 504 rows
└── split_metadata.json

outputs/
├── training/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── selection.json
└── evaluation/
    └── test_metrics.json
```

---

## Key Design Decisions (Quyết định thiết kế quan trọng)

1. **Chỉ dùng X, Y (không Z)**: Webcam 2D không đáng tin cậy về depth
2. **Padding trước khi detect**: Cải thiện detection rate cho ảnh bị crop sát
3. **normalize theo wrist + MCP9**: Bất biến vị trí + tỉ lệ, nhưng giữ orientation
4. **Logistic Regression thay vì deep learning**: 42 features đã được engineer tốt, không cần NN
5. **Grouped split**: Tránh data leakage khi cùng nguồn ảnh vào cả train lẫn test
6. **Hash check preprocessing.py**: Đảm bảo model và inference dùng đúng cùng preprocessing

---

*Tạo ngày 2026-10-05*

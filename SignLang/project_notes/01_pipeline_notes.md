# 📝 Ghi Chú Pipeline — Sign Language Recognition
> Mục đích: Tài liệu cho phần báo cáo. Giải thích TỪNG BƯỚC pipeline theo thứ tự thực tế.

---

## Tổng quan Pipeline

```
Webcam Frame
    │
    ▼
[Bước 1] Lấy ảnh từ webcam (OpenCV)
    │
    ▼
[Bước 2] Thêm padding đen vào viền ảnh (25% mỗi cạnh)
    │
    ▼
[Bước 3] Phát hiện bàn tay bằng MediaPipe Hand Landmarker
    │
    ▼
[Bước 4] Lấy 21 điểm đặc trưng (landmarks) từ kết quả
    │
    ▼
[Bước 5] Chuẩn hóa tọa độ — vị trí (wrist = gốc) + tỉ lệ (khoảng cách wrist→MCP9 = 1)
    │
    ▼
[Bước 6] Vector 42 features → Logistic Regression → nhãn ký hiệu (0-9, A-Z)
    │
    ▼
[Bước 7] Smoothing (trung bình 5 frame) → hiển thị kết quả cuối
```

---

## Bước 1: Lấy ảnh từ webcam

**File liên quan**: `11_webcam.py` (dòng 44-46)

```python
success, frame = capture.read()  # OpenCV đọc 1 frame từ webcam
```

**Tại sao**:
- OpenCV cung cấp API đơn giản để đọc webcam theo từng frame
- Frame là ảnh BGR (Blue-Green-Red) — format mặc định của OpenCV
- Chụp ở 640×480 pixels (dòng 32-33 trong `11_webcam.py`)
- Frame KHÔNG được mirror trước khi vào pipeline — mirror chỉ dùng để hiển thị

**Chi tiết kỹ thuật**:
- `cv2.VideoCapture(0)` — mở webcam index 0 (webcam mặc định)
- `--backend dshow` — Windows DirectShow, giải quyết vấn đề tương thích trên một số máy
- FPS được tính theo exponential moving average: `fps = 0.85*fps + 0.15*current_fps`

---

## Bước 2: Thêm padding đen (Black Padding)

**File liên quan**: `preprocessing.py` → hàm `add_image_padding()`

```python
def add_image_padding(image_rgb):
    pad_x = max(1, round(width * 0.25))   # 25% width mỗi bên trái/phải
    pad_y = max(1, round(height * 0.25))  # 25% height mỗi bên trên/dưới
    padded = np.pad(image_rgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 
                    mode="constant", constant_values=0)  # 0 = màu đen
    return padded, pad_x, pad_y
```

**Tại sao cần padding**:
- Ảnh từ dataset thường bị crop sát bàn tay → ngón tay nằm gần viền
- MediaPipe bị giảm hiệu suất phát hiện khi đặc trưng quá gần cạnh ảnh
- Thêm 25% padding mỗi cạnh → bàn tay "lùi vào giữa" → MediaPipe phát hiện tốt hơn
- Thực nghiệm (step 05): tỉ lệ phát hiện tăng từ ~94% lên ~95.7% sau padding
- Ảnh 400×400 → 600×600 sau padding

**Lưu ý quan trọng**:
- `pad_x`, `pad_y` được trả về để SAU đó có thể map điểm landmark về tọa độ ảnh gốc
- Padding chỉ dùng cho input của MediaPipe, KHÔNG ảnh hưởng feature vector cuối cùng

---

## Bước 3: Phát hiện bàn tay (Hand Detection)

**File liên quan**: `inference.py` (dòng 41-48), dùng model `models/hand_landmarker.task`

```python
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(task_path)),
    running_mode=vision.RunningMode.IMAGE,  # Độc lập từng frame
    num_hands=2,           # Cho phép phát hiện tối đa 2 tay
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)
result = detector.detect(mp.Image(...))
```

**Tại sao dùng MediaPipe Hand Landmarker**:
- Google's pre-trained model, được huấn luyện trên dataset lớn
- Phát hiện bàn tay + 21 điểm landmark trong 1 lần gọi
- Nhanh, chính xác, chạy được real-time
- File model: `hand_landmarker.task` (~7.8 MB)

**Logic chọn tay**:
- Nếu phát hiện **0 tay** → trả về `"no_hand"` → hiển thị "Show one hand clearly"
- Nếu phát hiện **>1 tay** → trả về `"multiple_hands"` → hiển thị "Keep only one hand visible"
- Nếu phát hiện **đúng 1 tay** → tiếp tục pipeline

**Tại sao `running_mode=IMAGE` không phải VIDEO**:
- Mode IMAGE xử lý từng frame độc lập → phù hợp với dataset ảnh tĩnh
- Mode VIDEO có tracking giữa frames → có thể drift nếu hand detection tạm thất bại
- Để nhất quán giữa training và inference, dùng cùng mode IMAGE

---

## Bước 4: Lấy 21 điểm landmark

**File liên quan**: `inference.py` (dòng 65)

```python
points = np.asarray([[p.x, p.y] for p in result.hand_landmarks[0]], dtype=np.float64)
# Kết quả: array shape (21, 2) — 21 điểm, mỗi điểm có (x, y)
```

**MediaPipe trả về tọa độ normalized** (0.0 → 1.0 relative to image size):
- `p.x = pixel_x / image_width`
- `p.y = pixel_y / image_height`

**21 điểm theo thứ tự MediaPipe**:
```
 0: Wrist (cổ tay)
 1-4: Ngón cái (thumb): CMC, MCP, IP, TIP
 5-8: Ngón trỏ (index): MCP, PIP, DIP, TIP
 9-12: Ngón giữa (middle): MCP, PIP, DIP, TIP
13-16: Ngón áp (ring): MCP, PIP, DIP, TIP
17-20: Ngón út (pinky): MCP, PIP, DIP, TIP
```

**Tại sao chỉ dùng X và Y (không dùng Z)**:
- Z là độ sâu — không ổn định với webcam 2D thông thường
- X, Y đủ để phân biệt các ký hiệu tay tĩnh (static gestures)
- 21 điểm × 2 tọa độ = 42 features

---

## Bước 5: Chuẩn hóa (Normalization)

**File liên quan**: `preprocessing.py` → hàm `normalize_landmarks()`

```python
def normalize_landmarks(landmarks_xy, image_width, image_height):
    points_pixels = landmarks_xy * [image_width, image_height]  # → pixel units
    centered = points_pixels - points_pixels[0]                  # → wrist = (0,0)
    scale = np.linalg.norm(centered[9])                          # khoảng cách wrist→MCP9
    normalized = centered / scale                                 # → tỉ lệ chuẩn hóa
    return normalized.reshape(42).astype(np.float32)             # → vector 42 chiều
```

**2 bước chuẩn hóa**:

### 5a. Chuẩn hóa vị trí (Position Normalization)
- **Vấn đề**: Bàn tay có thể ở bất kỳ vị trí nào trong frame
- **Giải pháp**: Lấy điểm Wrist (landmark #0) làm gốc tọa độ → trừ tất cả điểm đi
- **Kết quả**: Wrist luôn = (0, 0), các điểm khác là tọa độ tương đối

### 5b. Chuẩn hóa tỉ lệ (Scale Normalization)
- **Vấn đề**: Bàn tay có thể to/nhỏ tùy khoảng cách camera
- **Giải pháp**: Dùng khoảng cách từ Wrist (#0) đến Middle Finger MCP (#9) làm đơn vị
- **Kết quả**: Khoảng cách này luôn = 1.0, mọi tỉ lệ đều nhất quán

**Tại sao chọn điểm #9 làm reference**:
- MCP của ngón giữa là điểm ổn định nhất, ít bị ảnh hưởng bởi cử động ngón tay
- Khoảng cách wrist→MCP luôn dương và đủ lớn để làm đơn vị

**Tính chất bất biến** (invariance) đã được verify:
- ✅ Bất biến với vị trí (hand ở góc nào cũng cho ra feature vector như nhau)
- ✅ Bất biến với kích thước (hand gần/xa camera không ảnh hưởng)
- ❌ Không bất biến với rotation (xoay tay 90° sẽ cho feature khác — đây là chủ ý)
- ❌ Không bất biến với mirror (tay trái/phải cho feature khác — đây là chủ ý)

---

## Bước 6: Phân loại (Classification)

**File liên quan**: `inference.py` (dòng 74), model: `models/sign_classifier.joblib`

```python
probabilities = self.model.predict_proba(features.reshape(1, -1))[0]  # (36,) vector xác suất
winner = int(probabilities.argmax())   # Index của class cao nhất
label = str(self.classes[winner])      # Nhãn: "0"-"9", "A"-"Z"
score = float(probabilities[winner])   # Xác suất 0.0-1.0
```

**Model được chọn**: Logistic Regression (với StandardScaler)

**So sánh 3 model đã thử**:
| Model | Val Accuracy | Val Macro F1 | Lý do chọn/không chọn |
|-------|-------------|-------------|----------------------|
| Decision Tree | 75.15% | 0.7278 | Đơn giản, dễ overfit |
| Random Forest | 85.01% | 0.8310 | Tốt nhưng không phải best |
| **Logistic Regression** | **90.93%** | **0.8855** | ✅ Chọn — F1 cao nhất |

**Tại sao Logistic Regression tốt hơn**:
- 42 features đã được chuẩn hóa tốt → linear boundary đủ mạnh
- StandardScaler đưa features về mean=0, std=1 → LR hoạt động tốt
- Dataset không quá phức tạp với 36 class tĩnh

**Kết quả test cuối** (trên test set chưa từng thấy):
- Accuracy: 77.58% (thấp hơn validation vì test harder?)
- Macro F1: 0.7672

---

## Bước 7: Smoothing (làm mượt dự đoán)

**File liên quan**: `inference.py` → class `PredictionSmoother`

```python
class PredictionSmoother:
    def __init__(self, window=5, minimum_frames=3):
        self.history = deque(maxlen=window)  # giữ 5 frame gần nhất
    
    def update(self, probabilities):
        self.history.append(probabilities)
        if len(self.history) < self.minimum_frames:  # cần ít nhất 3 frame
            return None
        return np.mean(np.stack(self.history), axis=0)  # trung bình xác suất
```

**Tại sao cần smoothing**:
- Webcam có nhiễu → dự đoán frame-by-frame có thể nhảy lung tung
- Trung bình 5 frame → loại bỏ nhiễu, kết quả ổn định hơn
- Reset khi mất tay khỏi frame → tránh dự đoán cũ ảnh hưởng frame mới

**Threshold hiển thị**:
- Mặc định `--threshold 0.5` — chỉ hiển thị nhãn khi smoothed score ≥ 0.5
- Nếu score < 0.5 → hiển thị "Uncertain | Best guess: X (0.XX)"
- Threshold này KHÔNG được dùng để chọn model (tránh data leakage)

---

## Ghi chú về Data Leakage Prevention

Project này áp dụng nhiều biện pháp tránh data leakage:
1. **Test set chỉ dùng 1 lần** — sau khi chọn model xong mới evaluate
2. **Model selection dùng validation set** — không nhìn vào test set
3. **Grouped split** — ảnh từ cùng nguồn (same person/session) không bị split giữa train/test
4. **Threshold không được tuned** — giá trị 0.5 mặc định, không dùng test data để chọn

---

*File này được tạo ngày 2026-10-05. Cập nhật khi có thay đổi.*

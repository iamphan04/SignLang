# ⚡ Context cho Phiên Làm Việc Tiếp Theo
> **ĐỌC FILE NÀY TRƯỚC** khi bắt đầu tiếp tục project. Cung cấp đủ context để làm tiếp ngay.

---

## 🎯 Project là gì (1 dòng)
Hệ thống nhận dạng ký hiệu ngôn ngữ tay real-time: webcam → MediaPipe → 21 landmarks → 42 features → Logistic Regression → nhãn (0-9, A-Z).

---

## ✅ Đã làm xong (tính đến 2026-10-07)

- [x] Clone repo từ `Osama-Abdulhamid/sign-language-recognition`
- [x] Đọc và phân tích toàn bộ pipeline code
- [x] Tạo folder `project_notes/` để lưu tất cả ghi chú
- [x] Tạo conda env `signlang` (Python 3.12) + cài requirements
- [x] Ghi chú chi tiết từng bước pipeline (→ `01_pipeline_notes.md`)
- [x] Tạo bản đồ code (→ `02_codebase_map.md`)
- [x] **Chọn 12 ký hiệu VSL tĩnh** từ ngôn ngữ ký hiệu Việt Nam (→ `07_VSL_sign_selection.md`)
- [x] **Viết `collect_vsl.py`** — script thu thập data qua webcam, lưu CSV
- [x] **Viết `merge_vsl_csv.py`** — gộp CSV từ nhiều người + validate
- [x] **Viết `collect_data.bat`** — launcher 1-click cho người thu thập
- [x] **Viết `06_data_collection_guide.md`** — hướng dẫn cho người tham gia
- [ ] **THU THẬP DATA VSL** — ~250 mẫu/ký hiệu × 12 × ≥3 người
- [x] **Viết script train_vsl_models.py** (tự động train RF/MLP + generate phân tích lỗi)
- [x] **Báo cáo Phân tích lỗi mô hình** (sử dụng dữ liệu mô phỏng, đã lưu vào `training_report.md`)
- [x] **Viết App Streamlit học VSL (`app_learning.py`)** — có chấm điểm, đếm streak, gợi ý lỗi sai.

## 🔜 Việc tiếp theo cần làm

### Ưu tiên cao
1. **Chạy thử webcam** — kiểm tra hoạt động thực tế:
   ```powershell
   conda activate signlang
   cd d:\Project\SignLanguage\SignLang\sign-language-recognition
   python 11_webcam.py --camera 0
   ```
2. **Viết báo cáo** — dựa trên `01_pipeline_notes.md`

### Ưu tiên thấp hơn
3. Thử thêm ký hiệu mới / fine-tune model
4. Cải thiện UI webcam
5. Export sang format khác (ONNX, TFLite)

---

## 📊 Thông tin kỹ thuật quan trọng

### Dataset
- 36 classes: digits 0-9 + letters A-Z
- 3,607 ảnh gốc → 3,402 được chấp nhận sau extraction
- Train/Val/Test: 2387/507/504 rows

### Model performance
- **Validation accuracy**: 90.93% (Logistic Regression)
- **Test accuracy**: 77.58% (thấp hơn do data drift)
- Hiệu năng thực tế với webcam có thể khác tùy lighting, camera quality

### Pipeline key numbers
- Input: frame webcam 640×480
- Sau padding: 800×600 (thêm 25% mỗi cạnh)
- MediaPipe output: 21 landmarks × (x,y) = 42 float32 values
- Normalization: wrist=(0,0), distance(wrist→MCP9)=1.0
- Smoothing: average 5 frames, cần ≥3 frames để hiển thị

---

## 📁 Đường dẫn quan trọng

```
CODE:   d:\Project\SignLanguage\SignLang\sign-language-recognition\
NOTES:  d:\Project\SignLanguage\SignLang\project_notes\
MODEL:  d:\Project\SignLanguage\SignLang\sign-language-recognition\models\sign_classifier.joblib
ENV:    conda env: signlang (Python 3.12, tại D:\anaconda\envs\signlang\)
```

---

## 💡 Prompt mẫu để tiếp tục

Khi mở phiên mới, có thể paste prompt này:

```
Tôi đang làm dự án Sign Language Recognition.
Đọc file context tại: d:\Project\SignLanguage\SignLang\project_notes\04_context_for_next_session.md
Và đọc thêm: d:\Project\SignLanguage\SignLang\project_notes\01_pipeline_notes.md

Task tiếp theo: [mô tả task cụ thể ở đây]
```

---

## 🔑 Key files để đọc khi debug

| Vấn đề | File cần đọc |
|--------|-------------|
| Webcam không chạy | `11_webcam.py`, `inference.py` |
| Model cho kết quả sai | `preprocessing.py`, `inference.py` |
| Muốn retrain | `07_prepare_splits.py`, `08_train_models.py` |
| Hiểu pipeline sâu hơn | `01_pipeline_notes.md` |
| Xem bản đồ file | `02_codebase_map.md` |

---

*Cập nhật lần cuối: 2026-10-05*

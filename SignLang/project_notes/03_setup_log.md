# 🔧 Setup Log — Môi trường & Cài đặt

---

## Thông tin hệ thống

- **OS**: Windows
- **Ngày setup**: 2026-10-05
- **Git path**: `C:\Users\sunyn\AppData\Local\GitHubDesktop\app-3.5.3\resources\app\git\cmd\git.exe`
- **Conda**: `D:\anaconda\Library\bin\conda.bat` (phiên bản 25.11.1)
- **Python mặc định**: Python 3.14.7 tại `C:\msys64\ucrt64\bin\python.exe` (KHÔNG DÙNG — mediapipe không hỗ trợ 3.14)

---

## Conda Environment: `signlang`

```powershell
# Tạo môi trường (đã làm ngày 2026-10-05)
conda create -n signlang python=3.12 -y

# Kết quả: Python 3.12.15 được cài vào D:\anaconda\envs\signlang\

# Cài requirements
conda activate signlang
cd d:\Project\SignLanguage\SignLang\sign-language-recognition
pip install -r requirements.txt

# Kết quả: Tất cả 29 packages cài thành công
```

### Packages đã cài (từ requirements.txt)
| Package | Version | Mục đích |
|---------|---------|----------|
| mediapipe | 1.0.1 | Hand landmark detection |
| opencv-contrib-python | 5.0.0.93 | Webcam + image processing |
| scikit-learn | 1.9.1 | Logistic Regression classifier |
| numpy | 2.5.3 | Array operations |
| pandas | 3.0.5 | Data loading |
| matplotlib | 3.11.2 | Visualization |
| joblib | 1.6.0 | Model serialization |
| scipy | 1.18.1 | Scientific computing |
| tqdm | 4.70.1 | Progress bars |

---

## Cách kích hoạt môi trường

```powershell
# Trong PowerShell:
conda activate signlang

# Hoặc chạy trực tiếp không activate:
conda run -n signlang python 11_webcam.py --camera 0
```

---

## Lệnh chạy webcam

```powershell
conda activate signlang
cd d:\Project\SignLanguage\SignLang\sign-language-recognition

# Option 1: Backend tự động (thường dùng)
python 11_webcam.py --camera 0

# Option 2: Backend DirectShow (nếu option 1 lỗi)
python 11_webcam.py --camera 0 --backend dshow

# Option 3: Backend Media Foundation
python 11_webcam.py --camera 0 --backend msmf

# Nếu có nhiều webcam, thử --camera 1, --camera 2, ...
```

### Arguments của `11_webcam.py`
| Arg | Default | Mô tả |
|-----|---------|-------|
| `--camera` | 0 | Index webcam |
| `--threshold` | 0.5 | Ngưỡng hiển thị nhãn |
| `--window` | 5 | Số frame để smooth |
| `--no-mirror` | False | Tắt mirror display |
| `--backend` | auto | auto/dshow/msmf |

---

## Lỗi thường gặp & Cách fix

### Lỗi: Không mở được webcam
```
RuntimeError: Cannot open camera
```
**Fix**: Thử `--camera 1` hoặc `--backend dshow`. Đóng các app đang dùng webcam (Zoom, Teams...).

### Lỗi: Model không tìm thấy
```
FileNotFoundError: Missing MediaPipe model: .../hand_landmarker.task
```
**Fix**: Đảm bảo đang chạy từ đúng thư mục. File phải tồn tại ở `models/hand_landmarker.task`.

### Lỗi: sign_classifier.joblib không khớp preprocessing
```
ValueError: The model and preprocessing.py differ
```
**Fix**: Không sửa `preprocessing.py`. Nếu cần train lại, chạy `run_from_07.py`.

### Lỗi: Python 3.14 không hỗ trợ mediapipe
**Fix**: Luôn dùng conda env `signlang` (Python 3.12).

---

## Cấu trúc thư mục project

```
d:\Project\SignLanguage\
├── SignLang\
│   ├── sign-language-recognition\    ← CODE GỐC (clone từ GitHub)
│   │   ├── models\
│   │   │   ├── hand_landmarker.task  (7.8 MB)
│   │   │   └── sign_classifier.joblib (14 KB)
│   │   ├── data\processed\splits\
│   │   ├── outputs\
│   │   ├── 11_webcam.py
│   │   ├── inference.py
│   │   ├── preprocessing.py
│   │   └── ...
│   └── project_notes\                ← GHI CHÚ (folder này)
│       ├── 00_README.md
│       ├── 01_pipeline_notes.md
│       ├── 02_codebase_map.md
│       ├── 03_setup_log.md          ← File này
│       └── 04_context_for_next_session.md
```

---

*Tạo ngày 2026-10-05*

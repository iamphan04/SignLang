# 📁 PROJECT_NOTES — Sign Language Recognition

> **Mục đích của folder này**: Lưu toàn bộ ghi chú kỹ thuật, phân tích pipeline, và context để bất kỳ lúc nào tiếp tục làm việc chỉ cần đọc folder này.

---

## 🗂️ Danh sách file trong folder này

| File | Nội dung |
|------|----------|
| `00_README.md` | File này — tổng quan và hướng dẫn |
| `01_pipeline_notes.md` | Ghi chú chi tiết từng bước pipeline (dành cho báo cáo) |
| `02_codebase_map.md` | Bản đồ code — file nào làm gì |
| `03_setup_log.md` | Log cài đặt môi trường, lệnh chạy |
| `04_context_for_next_session.md` | **ĐỌC FILE NÀY TRƯỚC** khi tiếp tục — tóm tắt nhanh |

---

## 🚀 Cách bắt đầu phiên làm việc tiếp theo

1. Đọc `04_context_for_next_session.md` — có tóm tắt project state và TODO list
2. Đọc file ghi chú cụ thể theo task cần làm
3. Code gốc tại: `d:\Project\SignLanguage\SignLang\sign-language-recognition\`

---

## 📌 Thông tin cơ bản

- **Repo gốc**: https://github.com/Osama-Abdulhamid/sign-language-recognition
- **Tác giả gốc**: Ziad Mostafa
- **Ngày clone**: 2026-10-05
- **Project path**: `d:\Project\SignLanguage\SignLang\sign-language-recognition\`
- **Notes path**: `d:\Project\SignLanguage\SignLang\project_notes\`
- **Conda env**: `signlang` (Python 3.12)

---

## ⚡ Lệnh chạy nhanh

```powershell
# Kích hoạt môi trường
conda activate signlang

# Chạy webcam (nhận ký hiệu tay real-time)
cd d:\Project\SignLanguage\SignLang\sign-language-recognition
python 11_webcam.py --camera 0

# Nếu webcam không mở được, thử backend dshow
python 11_webcam.py --camera 0 --backend dshow

# Phím tắt khi đang chạy webcam:
# Q hoặc Esc = thoát
# M = bật/tắt mirror (lật ảnh)
# R = reset smoothing
```

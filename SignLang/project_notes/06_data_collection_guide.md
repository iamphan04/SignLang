# 📋 Hướng Dẫn Thu Thập Dữ Liệu VSL (Ngôn ngữ Ký hiệu Việt Nam)
> Tài liệu này dành cho bạn VÀ những người khác sẽ giúp thu thập data.
> Mục tiêu: 250 mẫu/ký hiệu × 12 ký hiệu × ≥3 người = ~9,000 mẫu

---

## 📦 Yêu cầu trước khi bắt đầu

1. Máy tính đã cài đặt sẵn môi trường (hỏi người setup nếu chưa có)
2. Webcam hoạt động
3. Ánh sáng đủ sáng, tay rõ ràng

---

## 🚀 Cách chạy (cho người thu thập)

**Cách đơn giản nhất** — Double-click file:
```
d:\Project\SignLanguage\SignLang\project_notes\collect_data.bat
```
→ Nhập ID của bạn (p1, p2, p3...) → Enter → Bắt đầu

**Hoặc PowerShell:**
```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
cd d:\Project\SignLanguage\SignLang\sign-language-recognition
D:\anaconda\envs\signlang\python.exe collect_vsl.py --person p2 --target 250
```

---

## ⌨️ Phím bấm khi thu thập

| Phím | Tác dụng |
|------|---------|
| **SPACE** hoặc **S** | ✅ Lưu mẫu hiện tại |
| **F** hoặc **→** | Sang ký hiệu tiếp theo |
| **A** hoặc **←** | Về ký hiệu trước |
| **1** đến **9**, **0** | Nhảy nhanh tới ký hiệu (1=A, 2=B, ...) |
| **D** | Xoá mẫu cuối nếu lưu nhầm |
| **Q** / **Esc** | Lưu và thoát |

---

## ✋ 12 Ký Hiệu Cần Thu Thập

*(Tham khảo video dạy bảng chữ cái VSL trên mạng để rõ hơn)*

| # | Ký hiệu | Hướng dẫn cầm tay |
|---|---------|------------------|
| 1 | **A** | Nắm tay, ngón cái đặt thẳng sát bên hông ngón trỏ |
| 2 | **B** | Bàn tay mở, 4 ngón thẳng đứng sát nhau, ngón cái gập vào lòng bàn tay |
| 3 | **C** | Các ngón tay cong lại tạo thành hình chữ C |
| 4 | **D** | Ngón trỏ chỉ thẳng lên, các ngón khác và ngón cái chụm tạo hình tròn |
| 5 | **E** | Các ngón tay gập quặp lại, đầu ngón tay chạm vào gốc ngón tay |
| 6 | **I** | Nắm tay, chỉ có ngón út giơ thẳng lên |
| 7 | **L** | Ngón cái và ngón trỏ mở rộng tạo hình chữ L, 3 ngón còn lại gập |
| 8 | **M** | Nắm tay, 3 ngón tay (trỏ, giữa, áp út) đè lên trên ngón cái |
| 9 | **O** | Tất cả các ngón tay cong và chụm đầu lại vào ngón cái thành hình chữ O |
| 10 | **U** | Ngón trỏ và ngón giữa giơ thẳng và khép sát vào nhau |
| 11 | **V** | Ngón trỏ và ngón giữa giơ thẳng tạo hình chữ V (✌️) |
| 12 | **Y** | Nắm tay, chỉ có ngón cái và ngón út dang rộng ra 2 bên (dấu hiệu shaka) |

---

## 💡 Mẹo thu thập chất lượng cao

### Đa dạng hóa (QUAN TRỌNG — giúp model tổng quát hơn)

**Mỗi ký hiệu hãy thử:**
- 🔄 **3 góc bàn tay**: Thẳng, nghiêng trái 15°, nghiêng phải 15°
- 📏 **3 khoảng cách**: Gần camera (25cm), bình thường (50cm), xa (75cm)
- 💡 **2 ánh sáng**: Sáng bình thường, tối hơn (dùng đèn 1 bên)
- 🤚 **Cả 2 tay** nếu có thể (ký hiệu VSL thường thuận tay phải, nhưng thu thêm tay trái cũng ok)

### Không làm
- ❌ Đừng chụp khi bàn tay bị che khuất
- ❌ Đừng đặt tay quá sát viền khung hình
- ❌ Đừng thu quá nhanh liên tục (mỗi mẫu cách nhau ~0.5 giây)
- ❌ Đừng giữ nguyên 1 tư thế cho toàn bộ 250 mẫu

---

## 📁 File output

Mỗi người tạo 1 file riêng:
```
data/ksl_collected/
├── ksl_p1.csv    ← Người 1
├── ksl_p2.csv    ← Người 2
├── ksl_p3.csv    ← Người 3
└── ksl_merged.csv  ← Tổng hợp (tạo bằng merge_ksl_csv.py)
```

**Cấu trúc CSV** (47 cột):
```
person_id | label | label_name | session | timestamp | x0 | y0 | x1 | y1 | ... | x20 | y20
```
- 5 cột metadata + 42 cột features (21 landmarks × x,y)
- Features đã normalized: wrist=(0,0), distance(wrist→MCP9)=1.0

---

## 🔄 Sau khi thu thập xong

```powershell
# Gộp tất cả file CSV thành 1
$env:PYTHONUTF8="1"
cd d:\Project\SignLanguage\SignLang\sign-language-recognition
D:\anaconda\envs\signlang\python.exe merge_ksl_csv.py

# Kết quả: data/ksl_collected/ksl_merged.csv
# Sau đó chạy training pipeline (step 07-09)
```

---

## ⚠️ Về dataset công khai (KSL-Guide, KAIST)

| Dataset | Giấy phép | Phù hợp? |
|---------|----------|---------|
| KSL-Guide (KAIST) | Research only | ✅ Báo cáo học thuật ok, ❌ Commercial |
| AI Hub 한국수어 데이터 | CC BY | ✅ Miễn phí, cần cite |
| 국립국어원 수어영상 | Công khai | ✅ Tra cứu tham khảo |

> **Lưu ý**: Nếu dùng dataset công khai, phải ghi rõ nguồn trong báo cáo và kiểm tra giấy phép trước khi dùng cho sản phẩm thương mại.

---

*Tạo ngày 2026-10-07*

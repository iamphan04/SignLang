# 🤟 Chọn Ký Hiệu KSL — Korean Sign Language (한국수어)
> Ghi chú chọn lọc 12 ký hiệu tĩnh từ hệ thống Jimunja (지문자) — bảng tay chữ cái KSL chính thống.
> Nguồn tham khảo: Quốc lập Quốc ngữ viện (국립국어원) — https://sldict.korean.go.kr

---

## 1. Lý do chọn Jimunja (지문자) thay vì ký hiệu từ vựng

| Tiêu chí | Ký hiệu từ vựng | Jimunja (지문자) |
|---------|----------------|----------------|
| **Tính tĩnh** | Nhiều ký hiệu có chuyển động | Hầu hết là **static handshape** ✅ |
| **Số lượng class** | Hàng ngàn từ vựng | 14 phụ âm + 10 nguyên âm cơ bản ✅ |
| **Phù hợp ML** | Cần temporal model (RNN/LSTM) | Đủ với Logistic Regression / CNN ✅ |
| **Tài liệu** | Khó tra cứu đầy đủ | Có chuẩn quốc gia rõ ràng ✅ |
| **Ý nghĩa giáo dục** | Phụ thuộc ngữ cảnh | Dạy bảng chữ cái — nền tảng KSL ✅ |

---

## 2. Danh sách 12 ký hiệu được chọn

### Nhóm: 12 phụ âm cơ bản KSL (지문자 자음)

Chọn các phụ âm có **handshape rõ ràng, ít nhập nhằng**, bỏ qua một số phụ âm kép dễ nhầm lẫn.

| STT | Ký hiệu | Tên | Mô tả Handshape (từ nguồn chính thống) | Độ phân biệt |
|-----|---------|-----|----------------------------------------|-------------|
| 1 | **ㄱ** | 기역 | Chỉ duỗi ngón trỏ thẳng, các ngón còn lại nắm nhẹ. Ngón trỏ tạo góc gập giống hình ㄱ | ✅ Rõ |
| 2 | **ㄴ** | 니은 | Duỗi ngón trỏ + ngón cái tạo hình chữ L (góc vuông), các ngón khác nắm | ✅ Rõ |
| 3 | **ㄷ** | 디귿 | Ngón cái + ngón trỏ tạo hình vuông chữ ㄷ, ngón giữa hỗ trợ | ⚠️ Dễ nhầm ㄴ |
| 4 | **ㄹ** | 리을 | Ngón trỏ + ngón giữa duỗi và uốn cong nhẹ, mô phỏng đường cong ㄹ | ✅ Rõ |
| 5 | **ㅁ** | 미음 | Ngón cái, trỏ, giữa chạm đầu nhau tạo hình vuông/hình chữ nhật minh họa ㅁ | ✅ Rõ |
| 6 | **ㅂ** | 비읍 | 4 ngón tay (trừ cái) chụm lại, hướng lên, mô phỏng 2 đường thẳng đứng của ㅂ | ✅ Rõ |
| 7 | **ㅅ** | 시옷 | Ngón trỏ + ngón giữa duỗi ra, mở tạo hình chữ V/ㅅ | ✅ Rõ |
| 8 | **ㅇ** | 이응 | Ngón cái + ngón trỏ tạo vòng tròn (OK sign), các ngón khác cong | ✅ Rõ |
| 9 | **ㅈ** | 지읒 | Ngón cái + ngón trỏ + ngón giữa tạo hình giống ㅈ | ✅ Rõ |
| 10 | **ㅎ** | 히읗 | Bàn tay mở rộng, ngón cái và ngón út dang ra, tạo hình tròn ở trên | ✅ Rõ |
| 11 | **ㅏ** | 아 (nguyên âm) | Ngón cái duỗi thẳng bên phải bàn tay nắm — nguyên âm đơn giản nhất | ✅ Rõ |
| 12 | **ㅣ** | 이 (nguyên âm) | Ngón trỏ duỗi thẳng đứng, các ngón còn lại nắm lại | ✅ Rõ |

---

## 3. Phân tích từng ký hiệu chi tiết (cho báo cáo)

### ㄱ (Giyeok)
- **Nguồn**: 국립국어원 한국수어사전, mục "ㄱ 지문자"
- **Handshape**: Index finger extended, remaining fingers lightly curled (fist-like)
- **Orientation**: Palm facing out or inward depending on signer preference
- **Ghi chú ML**: Đơn giản, ít ngón tay tham gia → dễ phân biệt qua 21 landmark

### ㄴ (Nieun)
- **Handshape**: Thumb (horizontal) + Index finger (vertical) → L-shape
- **Tương đồng với**: ASL letter "L"
- **Ghi chú ML**: L-shape rõ → landmark angles dễ học

### ㄷ (Digeut)
- **Handshape**: Thumb + Index + Middle tạo hình chữ C/ㄷ
- **⚠️ Dễ nhầm với**: ㄴ (chỉ khác vị trí ngón giữa)
- **Ghi chú ML**: Cần chú ý feature của landmark 7 (PIP ngón giữa)

### ㄹ (Rieul)
- **Handshape**: Index + Middle finger extended, slightly curled to form wave
- **Ghi chú ML**: Độ uốn ngón tay là đặc trưng quan trọng (DIP joint angles)

### ㅁ (Mieum)
- **Handshape**: Thumb tip touches index + middle tips → rectangular shape
- **Liên tưởng**: Miệng đóng lại = âm "m" → minh họa hình vuông ㅁ
- **Ghi chú ML**: Cluster 3 ngón → landmark 4, 8, 12 gần nhau

### ㅂ (Bieup)
- **Handshape**: 4 ngón (trừ cái) chụm thẳng lên trên, cái gập vào lòng bàn tay
- **Ghi chú ML**: 4 landmarks (5,9,13,17) gần nhau trên trục Y

### ㅅ (Siot)
- **Handshape**: Index + Middle duỗi ra tạo hình V (Peace sign) nhưng hướng xuống
- **Liên tưởng**: 2 đường chéo của ㅅ
- **⚠️ Dễ nhầm với**: ㄴ nếu orientation sai

### ㅇ (Ieung)
- **Handshape**: Thumb + Index tạo vòng tròn (giống "OK"), các ngón còn lại hơi cong
- **Liên tưởng**: Hình tròn ㅇ
- **Ghi chú ML**: Distance(landmark 4, landmark 8) ≈ 0 là feature quan trọng

### ㅈ (Jieut)
- **Handshape**: Thumb + Index + Middle tips gặp nhau ở điểm, tạo hình ㅈ
- **Ghi chú ML**: Tương tự ㅅ nhưng có ngón giữa tham gia

### ㅎ (Hieut)
- **Handshape**: Bàn tay mở, ngón cái và các ngón dang ra tạo hình tròn lớn
- **Ghi chú ML**: Nhiều ngón duỗi → landmark spread rộng nhất

### ㅏ (Vowel A)
- **Handshape**: Nắm tay, chỉ ngón cái duỗi sang phải
- **Liên tưởng**: Nét ngắn ngang của ㅏ
- **Ghi chú ML**: Chỉ landmark 4 xa, còn lại tụ lại

### ㅣ (Vowel I)
- **Handshape**: Chỉ ngón trỏ duỗi thẳng đứng, tất cả ngón khác nắm
- **Liên tưởng**: Nét thẳng đứng của ㅣ
- **Ghi chú ML**: Tương tự ㄱ nhưng ngón trỏ thẳng hơn, không gập

---

## 4. Lý do KHÔNG chọn một số ký hiệu

| Ký hiệu | Lý do bỏ qua |
|---------|-------------|
| **ㄲ, ㄸ, ㅃ, ㅆ, ㅉ** | Phụ âm kép — thường là động tác đôi (double tap) → KHÔNG static |
| **ㅗ, ㅜ, ㅛ, ㅠ** | Nguyên âm phức — cần 2 tay hoặc chuyển động → không phù hợp |
| **ㅐ, ㅔ** | Nguyên âm kép — dễ nhầm nhau, cần nhiều data |
| **Số 1-10 KSL** | Có biến thể địa phương → không nhất quán cho training |

---

## 5. Tóm tắt bộ 12 class cho model mới

```python
KSL_LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ", "ㅏ", "ㅣ"]
# 12 classes — 10 phụ âm cơ bản + 2 nguyên âm đơn giản nhất
```

**Kỳ vọng accuracy**: Với 12 class phân biệt rõ, Logistic Regression trên 42 features có thể đạt >85% nếu dataset đủ lớn (~100-200 ảnh/class).

---

## 6. Kế hoạch thu thập dữ liệu

1. Thu thập ảnh webcam: ~100-150 ảnh/class × 12 class = ~1400 ảnh
2. Môi trường đa dạng: nền sáng/tối, lighting khác nhau
3. Người thực hiện: ít nhất 3-5 người khác nhau để tránh overfitting
4. Kiểm tra với nguồn: so sánh với video trên sldict.korean.go.kr

---

## 7. Nguồn tài liệu chính thống

| Nguồn | URL | Nội dung |
|-------|-----|---------|
| 국립국어원 한국수어사전 | https://sldict.korean.go.kr | **Chính thống nhất** — có video từng ký hiệu |
| Encyclopedia of Korean Culture | https://aks.ac.kr | Giải thích lịch sử Jimunja |
| Research (CANKS) | canks.asia | Phân tích ML trên KSL consonants |
| Kaggle KSL Dataset | kaggle.com | Dataset tham khảo |

---

*Tạo ngày 2026-10-05. Dựa trên tài liệu chính thống 국립국어원.*

# 🤟 Chọn Ký Hiệu VSL — Vietnamese Sign Language (Ngôn ngữ Ký hiệu Việt Nam)
> Ghi chú chọn lọc 12 ký hiệu tĩnh từ bảng chữ cái VSL.
> Bảng chữ cái VSL chịu ảnh hưởng nhiều từ bảng chữ cái quốc tế/ASL, kết hợp với các dấu thanh đặc thù, nhưng ở đây ta chọn 12 chữ cái cơ bản nhất, hoàn toàn tĩnh (static) để phù hợp cho model Machine Learning (Logistic Regression).

---

## 1. Lý do chọn các chữ cái cơ bản VSL

| Tiêu chí | Đặc điểm được chọn |
|---------|----------------|
| **Tính tĩnh** | Hầu hết các chữ cái Latin cơ bản trong VSL là **static handshape** ✅ |
| **Độ phân biệt** | Tránh các chữ quá giống nhau ở các góc nhìn, chọn 12 chữ khác biệt rõ ràng ✅ |
| **Phù hợp ML** | Không cần mô hình temporal (như LSTM) vì không có chuyển động (dynamic) ✅ |

---

## 2. Danh sách 12 ký hiệu được chọn (VSL / ASL base)

| STT | Ký hiệu | Mô tả Handshape | Đặc trưng ML (Landmarks) |
|-----|---------|----------------------------------------|---------------------------|
| 1 | **A** | Nắm tay, ngón cái đặt thẳng sát bên hông ngón trỏ | Landmark 4 nằm sát LM5, các ngón khác gập |
| 2 | **B** | Bàn tay mở, 4 ngón thẳng đứng sát nhau, ngón cái gập vào lòng bàn tay | 4 ngón thẳng, LM4 gập vào trong |
| 3 | **C** | Các ngón tay cong lại tạo thành hình chữ C | Độ cong đều ở tất cả các ngón |
| 4 | **D** | Ngón trỏ chỉ thẳng lên, các ngón khác và ngón cái chụm tạo hình tròn | LM8 cao nhất, LM4 chạm LM12/16/20 |
| 5 | **E** | Các ngón tay gập quặp lại, đầu ngón tay chạm vào gốc ngón tay | Các đầu ngón tay cụp sát xuống lòng bàn tay |
| 6 | **I** | Nắm tay, chỉ có ngón út giơ thẳng lên | LM20 cao nhất, các ngón khác gập |
| 7 | **L** | Ngón cái và ngón trỏ mở rộng tạo hình chữ L, 3 ngón còn lại gập | Góc L rõ ràng giữa ngón cái và ngón trỏ |
| 8 | **M** | Nắm tay, 3 ngón tay (trỏ, giữa, áp út) đè lên trên ngón cái | Vị trí ngón cái bị kẹp dưới 3 ngón |
| 9 | **O** | Tất cả các ngón tay cong và chụm đầu lại vào ngón cái thành hình chữ O | Dist(LM4, LM8/12/16/20) ≈ 0 |
| 10 | **U** | Ngón trỏ và ngón giữa giơ thẳng và khép sát vào nhau | LM8 và LM12 sát nhau, các ngón khác gập |
| 11 | **V** | Ngón trỏ và ngón giữa giơ thẳng tạo hình chữ V (✌️) | Góc chữ V rộng giữa LM8 và LM12 |
| 12 | **Y** | Nắm tay, chỉ có ngón cái và ngón út dang rộng ra 2 bên (dấu hiệu shaka) | Khoảng cách xa nhất giữa LM4 và LM20 |

---

## 3. Tóm tắt bộ 12 class cho model mới

```python
VSL_LABELS = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]
```

---

## 4. Kế hoạch thu thập dữ liệu

1. Thu thập ảnh webcam: ~250 ảnh/class × 12 class = ~3000 ảnh / người.
2. Môi trường đa dạng: nền sáng/tối, góc nghiêng bàn tay.
3. Người thực hiện: ≥ 3 người khác nhau.

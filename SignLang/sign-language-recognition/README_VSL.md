# 🤟 Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam (VSL) Real-Time

Một hệ thống học và nhận diện ngôn ngữ ký hiệu Việt Nam (VSL) dạng tĩnh theo thời gian thực. Hệ thống sử dụng Camera, MediaPipe để trích xuất khung xương bàn tay, và Multi-Layer Perceptron (Neural Network) để phân loại.

> Đồ án tập trung vào 12 chữ cái VSL cơ bản dạng tĩnh: `A, B, C, D, E, I, L, M, O, U, V, Y`.

![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Placeholder:+Screenshot+of+Streamlit+App)

## 📌 Tính Năng Nổi Bật
- **Real-Time Inference:** Nhận diện ký hiệu trực tiếp trên webcam với FPS cao.
- **Ứng Dụng Học Tập (Streamlit):** Gamification việc học VSL với hệ thống tính điểm, streak, và gợi ý lỗi sai theo thời gian thực.
- **Độ Thích Ứng Cao:** Chuẩn hóa Scale & Translation Invariance giúp nhận diện tốt dù tay to/nhỏ hay ở mọi vị trí trên khung hình.

---

## 🛠️ Cài Đặt (Installation)

Yêu cầu môi trường: **Python 3.12+**

```bash
# 1. Clone repository
git clone <your-repo-link>
cd sign-language-recognition

# 2. Tạo môi trường ảo và kích hoạt
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
pip install streamlit
```

---

## 🚀 Cách Sử Dụng (Usage)

Dự án cung cấp 2 chế độ chạy khác nhau:

### 1. Chạy Ứng Dụng Học Tập (Streamlit Web App)
Đây là giao diện tương tác tốt nhất cho người dùng cuối.
```bash
streamlit run app_learning.py
```

### 2. Chạy Thu Thập Dữ Liệu
Dùng để đóng góp thêm dữ liệu cho mô hình.
```bash
python collect_vsl.py --person p1 --target 250
```

---

## 🏗️ Cấu Trúc Pipeline

1. **Input:** OpenCV bắt frame từ Webcam (hoặc Ảnh).
2. **Padding:** Thêm 25% viền đen giúp MediaPipe bắt tốt hơn các bàn tay sát mép màn hình.
3. **Detection:** MediaPipe Hand Landmarker trích xuất 21 điểm (X, Y).
4. **Normalization:** 
   - Lấy cổ tay làm gốc (Translation invariance).
   - Chia cho chiều dài từ cổ tay tới gốc ngón giữa (Scale invariance).
5. **Classification:** MLP Model xử lý vector 42 chiều và đưa ra nhãn.
6. **Smoothing:** Lấy trung bình dự đoán trong 10-15 frame gần nhất để chống nhiễu (flickering).

---

## 📚 Nguồn Tham Khảo (References)
- **Kiến trúc ML tĩnh cơ sở:** [Osama-Abdulhamid/sign-language-recognition](https://github.com/Osama-Abdulhamid/sign-language-recognition)
- **Phát hiện bàn tay:** [Google MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- **Hệ thống ký hiệu VSL:** Tham khảo bảng chữ cái Ngôn ngữ ký hiệu Việt Nam tiêu chuẩn.

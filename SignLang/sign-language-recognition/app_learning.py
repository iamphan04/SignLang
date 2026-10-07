import streamlit as st
import cv2
import time
import random
import numpy as np
from pathlib import Path

# Đảm bảo import được inference từ folder hiện tại
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import logic bypass DLL
import types
class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "dummy.py"
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        class MockCD: pass
        return MockCD
sys.modules['sklearn.linear_model._cd_fast'] = MockModule('sklearn.linear_model._cd_fast')

from inference import SignRecognizer, PredictionSmoother, draw_landmarks

# -----------------------------------------
# CẤU HÌNH STREAMLIT
# -----------------------------------------
st.set_page_config(page_title="Học Ký Hiệu VSL", layout="wide", page_icon="🤟")

# CSS làm đẹp giao diện
st.markdown("""
<style>
.big-font {
    font-size: 150px !important;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin: 0px;
    padding: 0px;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

st.title("🤟 Ứng dụng Học Ngôn Ngữ Ký Hiệu VSL")

# -----------------------------------------
# KHỞI TẠO STATE
# -----------------------------------------
VSL_LABELS = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]

if 'target_sign' not in st.session_state:
    st.session_state.target_sign = random.choice(VSL_LABELS)
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

def toggle_cam():
    st.session_state.camera_active = not st.session_state.camera_active

# -----------------------------------------
# GIAO DIỆN CHÍNH
# -----------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Camera Nhận Diện")
    st.button("Bật / Tắt Camera", on_click=toggle_cam)
    frame_placeholder = st.empty()

with col2:
    st.markdown("### Bảng Điểm")
    c1, c2 = st.columns(2)
    c1.metric("Điểm (Score)", st.session_state.score)
    c2.metric("Chuỗi (Streak)", st.session_state.streak)
    
    st.markdown("---")
    st.markdown("### Ký hiệu cần làm:")
    st.markdown(f'<p class="big-font">{st.session_state.target_sign}</p>', unsafe_allow_html=True)
    
    hint_placeholder = st.empty()
    success_placeholder = st.empty()

# -----------------------------------------
# VÒNG LẶP WEBCAM (chỉ chạy khi bật)
# -----------------------------------------
if st.session_state.camera_active:
    # Khởi tạo mô hình
    @st.cache_resource
    def load_recognizer():
        return SignRecognizer(Path(__file__).resolve().parent)
    
    recognizer = load_recognizer()
    # Mượt hóa: Yêu cầu 10 khung hình liên tiếp giống nhau
    smoother = PredictionSmoother(window=15, minimum_frames=10)
    
    # Mở camera OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Không thể mở webcam. Vui lòng kiểm tra quyền hoặc đóng ứng dụng camera khác.")
        st.session_state.camera_active = False
        st.rerun()

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi đọc khung hình.")
            break
            
        # 1. Gọi model dự đoán
        result = recognizer.process(frame)
        mean_scores = smoother.update(result.probabilities if result.status == "ok" else None)
        
        # 2. Vẽ Landmark lên hình
        frame_drawn = draw_landmarks(frame, result.points_pixels)
        frame_drawn = cv2.flip(frame_drawn, 1) # Lật gương để dễ nhìn
        
        # 3. Logic chấm điểm & Gợi ý
        if mean_scores is not None:
            best_idx = int(mean_scores.argmax())
            pred_label = str(recognizer.classes[best_idx])
            score = float(mean_scores[best_idx])
            
            if score > 0.65: # Ngưỡng tự tin
                if pred_label == st.session_state.target_sign:
                    # ĐÚNG KÝ HIỆU
                    st.session_state.score += 10
                    st.session_state.streak += 1
                    
                    # Phát pháo hoa
                    st.balloons()
                    success_placeholder.success(f"🎉 CHÍNH XÁC! Bạn vừa hoàn thành chữ **{st.session_state.target_sign}**")
                    
                    # Chọn chữ mới
                    old_target = st.session_state.target_sign
                    new_target = old_target
                    while new_target == old_target:
                        new_target = random.choice(VSL_LABELS)
                    st.session_state.target_sign = new_target
                    
                    # Reset để tránh bắt liên tiếp
                    smoother.reset()
                    
                    # Dừng hình lại xíu cho user đọc thông báo
                    frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")
                    time.sleep(1.5)
                    st.rerun() # Refresh lại trang
                else:
                    # SAI KÝ HIỆU -> ĐƯA GỢI Ý
                    st.session_state.streak = 0 # Đứt chuỗi
                    hint_placeholder.error(f"💡 Gợi ý: Bạn đang làm giống chữ **{pred_label}**, hãy thử sửa tay lại thành **{st.session_state.target_sign}** nhé!")
            else:
                hint_placeholder.info("⏳ Vui lòng giữ yên tay trong vài giây...")
        else:
            if result.status == "no_hand":
                hint_placeholder.warning("⚠️ Không thấy bàn tay nào trong khung hình.")
            elif result.status == "multiple_hands":
                hint_placeholder.warning("⚠️ Vui lòng chỉ giơ 1 bàn tay.")
            else:
                hint_placeholder.info("⏳ Đang thu thập khung hình...")
            
        # 4. Hiển thị lên UI Streamlit
        frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
    cap.release()

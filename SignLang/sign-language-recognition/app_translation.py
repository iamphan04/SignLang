import cv2
import threading
import time
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request

# Cập nhật imports cho Dynamic Inference
from inference_dynamic import DynamicSignRecognizer, DynamicSentenceBuilder
from inference import draw_landmarks

app = Flask(__name__)

project_root = Path(__file__).resolve().parent
recognizer = DynamicSignRecognizer(project_root)
sentence_builder = DynamicSentenceBuilder(stable_frames=12)

version = 0

def gen_frames():
    global version
    
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(0)
    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = capture.read()
        if not success:
            break
            
        # 1. Trích xuất features và Dự đoán chuỗi động
        label, score, display_pts = recognizer.process(frame)
        
        # 2. Xây dựng câu (chỉ áp dụng nếu model đã được train và trả về kết quả)
        old_sentence = " ".join(sentence_builder.sentence)
        new_sentence = sentence_builder.update(label, score)
        
        if new_sentence != old_sentence:
            version += 1
            
        # 3. Vẽ bộ xương tay
        if display_pts:
            frame = draw_landmarks(frame, display_pts)
            
        frame = cv2.flip(frame, 1) # Mirror
        
        # Overlay trạng thái AI
        cv2.putText(frame, f"AI: {label} ({int(score*100)}%)", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 210, 250), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return jsonify({
        "text": " ".join(sentence_builder.sentence),
        "version": version
    })

@app.route('/action', methods=['POST'])
def action():
    global version
    data = request.json
    action_type = data.get('action')
    
    if action_type == 'clear':
        sentence_builder.clear()
    elif action_type == 'backspace':
        if sentence_builder.sentence:
            sentence_builder.sentence.pop()
    elif action_type == 'space':
        sentence_builder.sentence.append(" ")
        
    version += 1
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

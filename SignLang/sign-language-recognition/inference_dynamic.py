import cv2
import numpy as np
from collections import deque
from pathlib import Path
import joblib

from model_common import load_bundle
from preprocessing import add_image_padding, normalize_landmarks
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class DynamicSignRecognizer:
    def __init__(self, root_dir, sequence_length=30):
        self.root_dir = Path(root_dir)
        
        # Load MediaPipe Detector
        task_path = self.root_dir / "models" / "hand_landmarker.task"
        if not task_path.is_file():
            raise FileNotFoundError(f"Missing MediaPipe model: {task_path}")
            
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)
        
        # Load MLP Dynamic Model
        model_path = self.root_dir / "models" / "dynamic_classifier.joblib"
        self.model_loaded = False
        if model_path.is_file():
            try:
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.classes = bundle["classes"]
                self.seq_len = bundle.get("sequence_length", sequence_length)
                self.model_loaded = True
            except Exception as e:
                print(f"Lỗi load model: {e}")
        
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
    def extract_features(self, frame_bgr):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, pad_x, pad_y = add_image_padding(image_rgb)
        ph, pw = padded.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
        result = self.detector.detect(mp_img)
        count = len(result.hand_landmarks)

        features = np.zeros(84, dtype=np.float32)
        display_points_list = []
        
        if count > 0:
            for idx in range(count):
                handedness = result.handedness[idx][0].category_name
                pts_norm = np.asarray([[p.x, p.y] for p in result.hand_landmarks[idx]], dtype=np.float64)
                
                try:
                    hand_features = normalize_landmarks(pts_norm, pw, ph)
                    if hand_features.shape == (42,) and np.isfinite(hand_features).all():
                        if handedness == 'Left':
                            features[0:42] = hand_features
                        else:
                            features[42:84] = hand_features
                            
                        display_pts = pts_norm * np.array([pw, ph]) - np.array([pad_x, pad_y])
                        display_points_list.append(display_pts)
                except ValueError:
                    continue

        return features, display_points_list

    def process(self, frame_bgr):
        features, display_pts = self.extract_features(frame_bgr)
        self.frame_buffer.append(features)
        
        predicted_label = "Chưa có Model"
        score = 0.0
        
        if self.model_loaded and len(self.frame_buffer) == self.sequence_length:
            # Flatten buffer
            sequence_array = np.array(self.frame_buffer) # (30, 84)
            flattened = sequence_array.reshape(1, -1) # (1, 2520)
            
            probabilities = self.model.predict_proba(flattened)[0]
            winner = int(probabilities.argmax())
            predicted_label = str(self.classes[winner])
            score = float(probabilities[winner])
            
        return predicted_label, score, display_pts

    def close(self):
        self.detector.close()

class DynamicSentenceBuilder:
    def __init__(self, stable_frames=10):
        self.stable_frames = stable_frames
        self.sentence = []
        self.last_added_word = None
        self.history = deque(maxlen=stable_frames)
        
    def update(self, label, score):
        if score < 0.6 or not label or label == "Chưa có Model":
            self.history.append("NONE")
            return " ".join(self.sentence)
            
        self.history.append(label)
        
        # Nếu toàn bộ các frame gần đây đều cùng 1 nhãn
        if len(self.history) == self.stable_frames and all(x == label for x in self.history):
            if label == "REST":
                # Nghỉ tay -> cho phép thêm từ mới ở chu kỳ sau
                self.last_added_word = None
            else:
                if label != self.last_added_word:
                    self.sentence.append(label)
                    self.last_added_word = label
                    
        return " ".join(self.sentence)
        
    def clear(self):
        self.sentence = []
        self.last_added_word = None
        self.history.clear()

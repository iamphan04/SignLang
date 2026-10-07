import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path

# Local imports
from preprocessing import add_image_padding, normalize_landmarks

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "dynamic_vsl"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Từ vựng thông dụng trong giao tiếp (Word-level signs)
DYNAMIC_LABELS = ["XIN_CHAO", "CAM_ON", "TOI", "BAN", "YEU", "KHOE_KHONG", "REST"]

# Số frame chuẩn cho mỗi chuỗi cử chỉ
SEQUENCE_LENGTH = 30

def build_detector():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)

def process_frame(detector, frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, pad_x, pad_y = add_image_padding(image_rgb)
    ph, pw = padded.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
    result = detector.detect(mp_img)
    count = len(result.hand_landmarks)

    features = np.zeros(84, dtype=np.float32)
    display_points_list = []
    
    if count == 0:
        return features, display_points_list

    try:
        for idx in range(count):
            handedness = result.handedness[idx][0].category_name # 'Left' or 'Right'
            pts_norm = np.asarray([[p.x, p.y] for p in result.hand_landmarks[idx]], dtype=np.float64)
            
            hand_features = normalize_landmarks(pts_norm, pw, ph)
            if hand_features.shape != (42,) or not np.isfinite(hand_features).all():
                continue
                
            if handedness == 'Left':
                features[0:42] = hand_features
            else:
                features[42:84] = hand_features
                
            display_pts = pts_norm * np.array([pw, ph]) - np.array([pad_x, pad_y])
            display_points_list.append(display_pts)
            
    except ValueError:
        pass

    return features, display_points_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="p1")
    parser.add_argument("--label", choices=DYNAMIC_LABELS, required=True)
    parser.add_argument("--samples", type=int, default=30, help="Số video/mẫu cần thu thập")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_DIR / f"{args.label}_{args.person}.npy"
    
    detector = build_detector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    sequences = []
    
    print(f"\n[Thu thập cử chỉ ĐỘNG: {args.label}]")
    print("Mỗi mẫu kéo dài 30 khung hình (~1 giây).")
    print("Bấm SPACE để bắt đầu ghi 1 mẫu.")
    print("Bấm Q để thoát và lưu.\n")
    
    while len(sequences) < args.samples:
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        canvas = frame.copy()
        
        cv2.putText(canvas, f"Label: {args.label} | Saved: {len(sequences)}/{args.samples}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, "Press SPACE to start recording 1 sample", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
        cv2.imshow("Dynamic VSL Collector", canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Bắt đầu ghi 30 frames
            sequence_features = []
            for i in range(SEQUENCE_LENGTH):
                succ, f = cap.read()
                if not succ: break
                f = cv2.flip(f, 1)
                
                feat, pts_list = process_frame(detector, cv2.flip(f, 1)) # detector needs unmirrored for left/right accuracy or handle mirror
                
                sequence_features.append(feat)
                
                # Draw
                draw_f = f.copy()
                if pts_list:
                    for pts in pts_list:
                        for p in np.rint(pts).astype(int):
                            cv2.circle(draw_f, tuple(p), 4, (255, 0, 0), -1)
                
                cv2.putText(draw_f, f"RECORDING... {i+1}/{SEQUENCE_LENGTH}", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Dynamic VSL Collector", draw_f)
                cv2.waitKey(10)
                
            if len(sequence_features) == SEQUENCE_LENGTH:
                sequences.append(sequence_features)
                print(f"✅ Đã lưu mẫu thứ {len(sequences)}")
            time.sleep(0.5)

    if sequences:
        np_seq = np.array(sequences) # Shape: (samples, 30, 84)
        if save_path.exists():
            existing = np.load(save_path)
            np_seq = np.concatenate([existing, np_seq], axis=0)
        np.save(save_path, np_seq)
        print(f"Đã lưu thành công tại {save_path} (Total: {len(np_seq)} samples)")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

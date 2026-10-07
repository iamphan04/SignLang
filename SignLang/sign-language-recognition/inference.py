"""Shared still-image/webcam inference, using the exact extraction preprocessing."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_common import load_bundle
from preprocessing import add_image_padding, normalize_landmarks


@dataclass
class FrameResult:
    status: str
    detected_hands: int
    points_pixels: object = None
    features: object = None
    probabilities: object = None
    label: str = ""
    score: float = 0.0


class SignRecognizer:
    def __init__(self, root):
        # Lazy imports allow training and CSV checks without a webcam runtime.
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        root = Path(root)
        self.bundle = load_bundle(root)
        self.model = self.bundle["model"]
        self.classes = np.asarray(self.model.classes_, dtype=str)
        self.mp = mp
        task_path = root / "models" / "hand_landmarker.task"
        if not task_path.is_file():
            raise FileNotFoundError(f"Missing MediaPipe model: {task_path}")
        # IMAGE mode runs detection independently per frame, matching step 06.
        # A later tracking optimization would need its own local verification.
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        # Forest inference on one frame avoids launching worker threads every time.
        if "n_jobs" in self.model.get_params(deep=False):
            self.model.set_params(n_jobs=1)

    def process(self, image_bgr):
        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        padded, pad_x, pad_y = add_image_padding(image_rgb)
        height, width = padded.shape[:2]
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=padded))
        count = len(result.hand_landmarks)
        if count == 0:
            return FrameResult("no_hand", count)

        features = np.zeros(84, dtype=np.float32)
        display_points_list = []

        try:
            for idx in range(count):
                handedness = result.handedness[idx][0].category_name
                points = np.asarray([[p.x, p.y] for p in result.hand_landmarks[idx]], dtype=np.float64)
                
                hand_features = normalize_landmarks(points, width, height)
                if hand_features.shape != (42,) or not np.isfinite(hand_features).all():
                    raise ValueError("Expected 42 finite features per hand.")
                    
                if handedness == 'Left':
                    features[0:42] = hand_features
                else:
                    features[42:84] = hand_features
                    
                display_points = points * np.array([width, height]) - np.array([pad_x, pad_y])
                display_points_list.append(display_points)
                
        except ValueError:
            return FrameResult("invalid_landmarks", count)

        # Attempt to predict if the model supports 84 features.
        # Fallback to zeros/none if model expects 42 (until retrained)
        expected_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 42
        if expected_features == 42:
            # Fallback to original behavior to prevent crash with old models
            # Just take the first hand found or the first 42 features (Left hand)
            inference_features = features[0:42] if not np.all(features[0:42] == 0) else features[42:84]
        else:
            inference_features = features

        probabilities = np.asarray(self.model.predict_proba(inference_features.reshape(1, -1))[0], dtype=float)
        winner = int(probabilities.argmax())
        
        return FrameResult("ok", count, display_points_list, features, probabilities,
                           str(self.classes[winner]), float(probabilities[winner]))

    def close(self):
        self.detector.close()


class PredictionSmoother:
    """Average recent scores for display; clear stale predictions on invalid frames."""
    def __init__(self, window=5, minimum_frames=3):
        if not 1 <= minimum_frames <= window:
            raise ValueError("Expected 1 <= minimum_frames <= window.")
        self.history = deque(maxlen=window)
        self.minimum_frames = minimum_frames

    def reset(self):
        self.history.clear()

    def update(self, probabilities):
        if probabilities is None:
            self.reset()
            return None
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            self.reset()
            return None
        self.history.append(values.copy())
        if len(self.history) < self.minimum_frames:
            return None
        return np.mean(np.stack(self.history), axis=0)


def draw_landmarks(image_bgr, display_points_list):
    import cv2
    if not display_points_list:
        return image_bgr.copy()
    output = image_bgr.copy()
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
              (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (17, 0))
    for points_pixels in display_points_list:
        points = np.rint(points_pixels).astype(int)
        for chain in chains:
            for first, second in zip(chain, chain[1:]):
                cv2.line(output, tuple(points[first]), tuple(points[second]), (200, 220, 20), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(output, tuple(point), 3, (20, 230, 250), -1, cv2.LINE_AA)
            if index == 0:
                cv2.putText(output, "0", tuple(point + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def status_message(status):
    return {"no_hand": "Show hands clearly", 
            "invalid_landmarks": "Hand landmarks could not be normalized", "ok": "Hand(s) detected"}.get(status, status)

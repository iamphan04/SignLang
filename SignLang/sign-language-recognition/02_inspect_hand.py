"""Step 2: inspect MediaPipe hand landmarks on one dataset image.

Save this file directly inside sign_language_recognition.
Download the official Hand Landmarker model to models/hand_landmarker.task.
This script extracts raw image-relative x/y values. Project-specific position
and scale normalization, sign classification, and dataset-wide validation come
in later steps. A successful single-image result is not a dataset success rate.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


# 1. Choose exactly one image: the first filename in class B by default.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SELECTED_LABEL = "B"
IMAGE_INDEX = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class_folder = DATA_DIR / SELECTED_LABEL
if not class_folder.is_dir():
    raise FileNotFoundError(f"Class folder not found: {class_folder}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Download the Hand Landmarker model to: {MODEL_PATH}")

image_paths = []
for path in class_folder.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths.append(path)
image_paths.sort()

if not 0 <= IMAGE_INDEX < len(image_paths):
    raise ValueError(f"IMAGE_INDEX is out of range; found {len(image_paths)} images.")

image_path = image_paths[IMAGE_INDEX]
print(f"Selected image: {image_path}")
print(f"Folder label (not a prediction): {SELECTED_LABEL}")


# 2. Read the image and give MediaPipe an RGB image.
image_bgr = cv2.imread(str(image_path))
if image_bgr is None:
    raise ValueError(f"OpenCV could not read this image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
height, width = image_rgb.shape[:2]
print(f"Image dimensions: {width} x {height} pixels")
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


# 3. Configure the detector for a still image and at most one hand.
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

with vision.HandLandmarker.create_from_options(options) as detector:
    result = detector.detect(mp_image)

print(f"Detected hands (configured limit: 1): {len(result.hand_landmarks)}")


# 4. Prepare an original/annotated comparison, even if no hand was detected.
figure, axes = plt.subplots(1, 2, figsize=(11, 5))
for axis in axes:
    axis.imshow(image_rgb)
    axis.axis("off")
axes[0].set_title(f"Original image | folder label: {SELECTED_LABEL}")

if not result.hand_landmarks:
    axes[1].set_title("No hand detected in this image")
    print("No landmarks extracted. Share the saved preview to inspect this case.")
else:
    hand_landmarks = result.hand_landmarks[0]

    # Keep x and y only, in the fixed landmark order 0 through 20.
    coordinates = []
    for landmark in hand_landmarks:
        coordinates.append([landmark.x, landmark.y])

    points_xy = np.array(coordinates, dtype=np.float32)
    if points_xy.shape != (21, 2):
        raise ValueError(f"Unexpected landmark array shape: {points_xy.shape}")

    raw_features = points_xy.reshape(-1)
    print(f"Landmark array shape: {points_xy.shape}")
    print(f"Raw XY feature count: {raw_features.size}")
    print("Feature order: x0, y0, x1, y1, ..., x20, y20")
    print("Position and scale normalization have not been applied yet.")
    print("\nID          x          y")
    for index, (x, y) in enumerate(points_xy):
        print(f"{index:2d}  {x:9.6f}  {y:9.6f}")

    # Convert image-relative coordinates to pixels for visualization only.
    pixel_points = points_xy * np.array([width, height])
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = pixel_points[connection.start]
        end = pixel_points[connection.end]
        axes[1].plot(
            [start[0], end[0]], [start[1], end[1]],
            color="#16c784", linewidth=1.8, zorder=2,
        )

    axes[1].scatter(
        pixel_points[:, 0], pixel_points[:, 1],
        s=24, color="#ffca28", edgecolors="black", linewidths=0.5, zorder=3,
    )
    for index, (x, y) in enumerate(pixel_points):
        axes[1].annotate(
            str(index), (x, y), xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.5},
            zorder=4,
        )
    axes[1].set_title("21 hand landmarks | IDs 0-20")

# Keep both panels at the original image bounds.
for axis in axes:
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)

figure.suptitle(f"Hand landmark inspection | {image_path.name}")
figure.tight_layout(rect=(0, 0, 1, 0.95))
OUTPUT_DIR.mkdir(exist_ok=True)
preview_path = OUTPUT_DIR / f"hand_landmarks_{SELECTED_LABEL}_{IMAGE_INDEX}.png"
figure.savefig(preview_path, dpi=180)
print(f"\nSaved preview: {preview_path}")
print("Close the image window to finish the script.")
plt.show()
plt.close(figure)

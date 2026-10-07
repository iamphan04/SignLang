"""Step 1: count image files by label and display six sample images.

Save this file directly inside your sign_language_recognition project folder.
Counts are based on file extensions; this step does not validate every image
or check whether MediaPipe can detect a hand in it.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


# 1. Locate the dataset relative to this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

OUTPUT_DIR.mkdir(exist_ok=True)


# 2. Collect and sort the class folders.
class_folders = []

for path in DATA_DIR.iterdir():
    if path.is_dir():
        class_folders.append(path)

class_folders.sort()

if not class_folders:
    raise ValueError("The dataset folder contains no class folders.")


# 3. Count image files and remember one sample path for each class.
records = []
sample_paths = {}

for class_folder in class_folders:
    image_paths = []

    for path in class_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    image_paths.sort()
    label = class_folder.name
    image_count = len(image_paths)

    records.append({"label": label, "image_count": image_count})

    if image_paths:
        sample_paths[label] = image_paths[0]


# 4. Display the counts and save them as a CSV table.
summary = pd.DataFrame(records)
print(f"Dataset folder: {DATA_DIR}\n")
print(summary.to_string(index=False))
print(f"\nNumber of classes: {len(summary)}")
print(f"Total image files: {summary['image_count'].sum()}")

minimum = summary["image_count"].min()
maximum = summary["image_count"].max()
print(f"Smallest class: {minimum} image files")
print(f"Largest class: {maximum} image files")

if minimum == 0:
    print("Some classes have no matching image files.")
elif minimum == maximum:
    print("Class counts are exactly balanced.")
else:
    print(f"Class counts differ. Largest/smallest ratio: {maximum / minimum:.2f}")

summary_path = OUTPUT_DIR / "dataset_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\nSaved counts: {summary_path}")
print("Counts use file extensions; not every image has been decoded yet.")


# 5. Display the first image, by filename order, from six selected classes.
selected_labels = ["0", "1", "2", "A", "B", "C"]
figure, axes = plt.subplots(2, 3, figsize=(10, 7))

for axis, label in zip(axes.flat, selected_labels):
    axis.axis("off")
    axis.set_title(f"Label: {label}")
    image_path = sample_paths.get(label)

    if image_path is None:
        axis.text(0.5, 0.5, "No image files", ha="center", va="center")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read sample image: {image_path}")
        axis.text(0.5, 0.5, "Image unreadable", ha="center", va="center")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axis.imshow(image_rgb)

figure.suptitle("Dataset examples: one image per selected label")
figure.tight_layout(rect=(0, 0, 1, 0.96))
samples_path = OUTPUT_DIR / "sample_images.png"
figure.savefig(samples_path, dpi=160)
print(f"Saved examples: {samples_path}")
print("Close the image window to finish the script.")
plt.show()
plt.close(figure)

"""Extract the project's 42 normalized features plus a label from all images.

Place this file beside preprocessing.py in sign_language_recognition.
Every accepted row has provenance in image_audit.csv. Rejected images remain
in the audit and failure report. Exact pixel duplicates are reported and kept
for review before splitting; they must not cross train/validation/test splits.
No classifier is trained and no source images are changed by this script.
"""

from pathlib import Path
import hashlib
import string

import numpy as np
import pandas as pd

from preprocessing import normalize_landmarks


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "outputs"
EXPECTED_LABELS = list(string.digits + string.ascii_uppercase)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FAILURE_STATUSES = ["read_error", "no_hand", "multiple_hands", "invalid_landmarks"]
AUDIT_COLUMNS = [
    "source_image", "label", "status", "detail", "feature_row",
    "width", "height", "detected_hands", "handedness", "image_sha256",
]


def build_output_tables(feature_rows, audit_rows, class_labels):
    """Build the required CSV table, source mapping, and per-class report."""
    feature_columns = []
    for index in range(21):
        feature_columns.extend([f"x{index}", f"y{index}"])

    features = pd.DataFrame(feature_rows, columns=feature_columns + ["label"])
    features[feature_columns] = features[feature_columns].astype(np.float32)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    for column in ["feature_row", "width", "height", "detected_hands"]:
        audit[column] = pd.array(audit[column], dtype="Int64")

    # Verify the mapping after failed images have been skipped.
    accepted = audit.loc[audit["status"] == "ok"]
    if accepted["feature_row"].isna().any() or accepted["feature_row"].tolist() != list(range(len(features))):
        raise ValueError("Feature rows and their source-image records do not match.")
    if accepted["label"].tolist() != features["label"].tolist():
        raise ValueError("A feature row's label does not match its source image.")

    known_statuses = {"ok", *FAILURE_STATUSES}
    if not set(audit["status"]).issubset(known_statuses):
        raise ValueError("The image audit contains an unknown processing status.")

    summary_rows = []
    for label in class_labels:
        class_audit = audit.loc[audit["label"] == label]
        total = len(class_audit)
        successful = int((class_audit["status"] == "ok").sum())
        record = {
            "label": label,
            "total_images": total,
            "accepted": successful,
            "skipped": total - successful,
            "success_rate_pct": round(100 * successful / total, 2) if total else 0.0,
        }
        for status in FAILURE_STATUSES:
            record[status] = int((class_audit["status"] == status).sum())
        summary_rows.append(record)

    return features, audit, pd.DataFrame(summary_rows)


def main() -> None:
    # Image libraries are needed by extraction, not by the table checks above.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tqdm import tqdm

    # 1. Locate all the class folders and their image files.
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL_PATH}")

    found_labels = {path.name for path in DATA_DIR.iterdir() if path.is_dir()}
    if found_labels != set(EXPECTED_LABELS):
        missing = sorted(set(EXPECTED_LABELS) - found_labels)
        extra = sorted(found_labels - set(EXPECTED_LABELS))
        raise ValueError(f"Check class folders. Missing: {missing}; extra: {extra}")

    image_entries = []
    for label in EXPECTED_LABELS:
        paths = []
        for path in (DATA_DIR / label).rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
        for path in sorted(paths):
            image_entries.append((label, path))

    if not image_entries:
        raise ValueError("No image files were found in the dataset.")

    print(f"Dataset: {DATA_DIR}")
    print(f"Classes: {len(EXPECTED_LABELS)}")
    print(f"Image files to process: {len(image_entries)}")
    print("Accepting exactly one detected hand per image.")

    # 2. Create one detector and reuse it for all independent still images.
    # Allow detection of a second hand so ambiguous multi-hand images can be
    # reported instead of silently choosing one of them.
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    feature_rows = []
    audit_rows = []

    with vision.HandLandmarker.create_from_options(options) as detector:
        for label, image_path in tqdm(image_entries, desc="Extracting", unit="image"):
            # One audit record for every image, including unsuccessful images.
            record = {column: None for column in AUDIT_COLUMNS}
            record["source_image"] = image_path.relative_to(DATA_DIR).as_posix()
            record["label"] = label
            record["detail"] = ""
            audit_rows.append(record)

            # 3. Read one image at a time; keep only features and small records.
            # imdecode also supports Windows paths containing non-ASCII text.
            try:
                image_bytes = image_path.read_bytes()
                image_bgr = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
            except (OSError, cv2.error) as error:
                record["status"] = "read_error"
                record["detail"] = str(error)
                continue

            if image_bgr is None:
                record["status"] = "read_error"
                record["detail"] = "OpenCV could not decode this file."
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            record["width"], record["height"] = width, height

            # Same dimensions and exact RGB pixels produce the same fingerprint.
            # Changed pixels (such as JPEG recompression) need other checks.
            digest = hashlib.sha256()
            digest.update(str(image_rgb.shape).encode("ascii"))
            digest.update(image_rgb.tobytes())
            record["image_sha256"] = digest.hexdigest()

            # 4. Extract landmarks. Unexpected detector errors stop execution;
            # they are not silently converted into ordinary bad-image records.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = detector.detect(mp_image)
            hand_count = len(result.hand_landmarks)
            record["detected_hands"] = hand_count

            if hand_count == 0:
                record["status"] = "no_hand"
                record["detail"] = "No hand detected at the current thresholds."
                continue
            if hand_count > 1:
                record["status"] = "multiple_hands"
                record["detail"] = "More than one hand detected; needs review."
                continue

            coordinates = []
            for landmark in result.hand_landmarks[0]:
                coordinates.append([landmark.x, landmark.y])
            points_xy = np.array(coordinates, dtype=np.float64)

            # 5. Use the SAME normalization function as the worked example.
            try:
                features = normalize_landmarks(
                    points_xy, image_width=width, image_height=height
                )
                if features.shape != (42,) or not np.isfinite(features).all():
                    raise ValueError("Expected 42 finite normalized features.")
            except ValueError as error:
                record["status"] = "invalid_landmarks"
                record["detail"] = str(error)
                continue

            record["feature_row"] = len(feature_rows)  # Zero-based data row.
            record["status"] = "ok"
            if result.handedness and result.handedness[0]:
                record["handedness"] = result.handedness[0][0].category_name
            feature_rows.append(features.tolist() + [label])

    # 6. Build and validate the four output tables after the complete pass.
    features_table, audit_table, summary = build_output_tables(
        feature_rows, audit_rows, EXPECTED_LABELS
    )
    failures = audit_table.loc[audit_table["status"] != "ok"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_tables = [
        (features_table, PROCESSED_DIR / "features.csv"),
        (audit_table, REPORT_DIR / "image_audit.csv"),
        (summary, REPORT_DIR / "extraction_summary.csv"),
        (failures, REPORT_DIR / "failed_images.csv"),
    ]
    for table, path in output_tables:
        table.to_csv(path, index=False, encoding="utf-8-sig")

    print("\nPer-class extraction results:")
    print(summary.to_string(index=False))
    print(f"\nOriginal image files: {len(image_entries)}")
    print(f"Accepted images: {len(features_table)}")
    print(f"Skipped images: {len(failures)}")
    print(f"Feature CSV shape: {features_table.shape}")
    print(f"Overall extraction success: {100 * len(features_table) / len(image_entries):.2f}%")
    print("Extraction success measures usable rows, not classification accuracy.")

    # 7. Preserve duplicate information for review before any data split.
    accepted_audit = audit_table.loc[audit_table["status"] == "ok"]
    duplicate_rows = int(accepted_audit["image_sha256"].duplicated().sum())
    labels_per_hash = accepted_audit.groupby("image_sha256")["label"].nunique()
    conflicting_groups = int((labels_per_hash > 1).sum())
    print(f"Exact duplicate accepted rows beyond the first: {duplicate_rows}")
    print(f"Exact-image groups with conflicting labels: {conflicting_groups}")

    zero_classes = summary.loc[summary["accepted"] == 0, "label"].tolist()
    print(f"Classes with no accepted images: {zero_classes}")
    if zero_classes:
        print("Review those classes before training a 36-class classifier.")

    print("\nSaved files:")
    for _, path in output_tables:
        print(path)


if __name__ == "__main__":
    main()

"""Extract 42 features per image with the padding tested in step 05.

Place this file beside preprocessing.py in sign_language_recognition.
Use the updated preprocessing.py. Baseline results from step 03 are read for
comparison, while this run writes separate *_padded files. All images go through
the same padding and normalization. A numerical 'ok' remains a candidate row.
Rows flagged review_required must be excluded from training until the intended
hand is verified. Exact duplicates also need handling before any data split.
"""

from pathlib import Path
import hashlib
import json
import string

import numpy as np
import pandas as pd

from preprocessing import IMAGE_PADDING_FRACTION, add_image_padding, normalize_landmarks


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "outputs"
BASELINE_AUDIT_PATH = REPORT_DIR / "image_audit.csv"
EXPECTED_LABELS = list(string.digits + string.ascii_uppercase)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FAILURE_STATUSES = ["read_error", "no_hand", "multiple_hands", "invalid_landmarks"]
AUDIT_COLUMNS = [
    "source_image", "label", "status", "detail", "feature_row",
    "width", "height", "detected_hands", "handedness", "image_sha256",
    "padded_width", "padded_height", "pad_x", "pad_y", "review_required",
]


def build_output_tables(feature_rows, audit_rows, class_labels):
    """Build the required CSV table, source mapping, and per-class report."""
    feature_columns = []
    for index in range(21):
        feature_columns.extend([f"x{index}", f"y{index}"])

    features = pd.DataFrame(feature_rows, columns=feature_columns + ["label"])
    features[feature_columns] = features[feature_columns].astype(np.float32)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    for column in ["feature_row", "width", "height", "detected_hands",
                   "padded_width", "padded_height", "pad_x", "pad_y"]:
        audit[column] = pd.array(audit[column], dtype="Int64")
    audit["review_required"] = audit["review_required"].astype(bool)

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


def compare_with_baseline(audit, baseline, summary):
    """Match by source path, verify unchanged pixels, and count gains AND losses."""
    old = baseline[["source_image", "label", "status", "image_sha256"]].rename(columns={
        "label": "baseline_label", "status": "baseline_status", "image_sha256": "baseline_sha256",
    })
    comparison = audit.merge(old, on="source_image", how="outer", validate="one_to_one", indicator=True)
    if comparison["_merge"].ne("both").any():
        raise ValueError("The baseline and padded runs must cover the same image paths.")
    if comparison["label"].ne(comparison["baseline_label"]).any():
        raise ValueError("An image label changed since the baseline run.")
    old_hash = comparison["baseline_sha256"].fillna("")
    new_hash = comparison["image_sha256"].fillna("")
    both_readable = old_hash.ne("") & new_hash.ne("")
    if (both_readable & old_hash.ne(new_hash)).any():
        raise ValueError("Some image pixels changed since the baseline. Comparison would be invalid.")
    comparison = comparison.drop(columns=["_merge", "baseline_label", "baseline_sha256"])
    old_ok = comparison["baseline_status"].eq("ok")
    new_ok = comparison["status"].eq("ok")
    comparison["change"] = np.select(
        [old_ok & new_ok, ~old_ok & new_ok, old_ok & ~new_ok],
        ["kept_ok", "gained", "lost"], default="still_skipped",
    )

    changes = []
    for label, rows in comparison.groupby("label", sort=False):
        changes.append({
            "label": label,
            "baseline_accepted": int(rows["baseline_status"].eq("ok").sum()),
            "gained": int(rows["change"].eq("gained").sum()),
            "lost": int(rows["change"].eq("lost").sum()),
            "review_required": int(rows["review_required"].sum()),
        })
    summary = summary.merge(pd.DataFrame(changes), on="label", how="left", validate="one_to_one")
    change_columns = ["baseline_accepted", "gained", "lost", "review_required"]
    summary[change_columns] = summary[change_columns].fillna(0).astype(int)
    if not (summary["baseline_accepted"] + summary["gained"] - summary["lost"]).eq(summary["accepted"]).all():
        raise ValueError("The comparison totals do not reconcile.")
    return comparison, summary


def save_quality_preview(comparison, landmark_rows):
    """Show up to six newly accepted no_hand cases, sampled from distinct labels."""
    import cv2
    import matplotlib.pyplot as plt
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

    pool = comparison.loc[
        comparison["baseline_status"].eq("no_hand") & comparison["status"].eq("ok")
    ].sort_values("source_image")
    if pool.empty:
        return None
    representatives = pool.groupby("label", group_keys=False).sample(n=1, random_state=42)
    chosen = representatives.sample(n=min(6, len(representatives)), random_state=42)
    fig, axes = plt.subplots(2, 3, figsize=(12, 11), squeeze=False)
    fig.suptitle("Padded extraction | sample of newly detected hands", fontsize=16)
    fig.text(0.5, 0.93, "Previously no_hand; now passes single-hand checks. Inspect landmark placement.",
             ha="center", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, chosen.itertuples(index=False)):
        path = DATA_DIR / row.source_image
        bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Preview could not read: {path}")
        padded, _, _ = add_image_padding(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        height, width = padded.shape[:2]
        points = landmark_rows[int(row.feature_row)] * np.array([width, height])
        ax.imshow(padded)
        for edge in HandLandmarksConnections.HAND_CONNECTIONS:
            ax.plot(points[[edge.start, edge.end], 0], points[[edge.start, edge.end], 1],
                    color="#00e5ff", linewidth=1.4)
        ax.scatter(points[:, 0], points[:, 1], s=12, c="#ffcc33", edgecolors="black", linewidths=0.3)
        ax.annotate("0", points[0], xytext=(3, 3), textcoords="offset points", color="white", fontsize=10)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_title(f"Label: {row.label} | feature_row: {int(row.feature_row)}", fontsize=11)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.045, hspace=0.16, wspace=0.07)
    fig.text(0.5, 0.015, "feature_row refers to the zero-based data row in features_padded.csv.",
             ha="center", fontsize=9)
    destination = REPORT_DIR / "padded_quality_preview.png"
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    return destination


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
    if not BASELINE_AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Run step 03 first. Missing baseline audit: {BASELINE_AUDIT_PATH}")
    baseline = pd.read_csv(BASELINE_AUDIT_PATH,
                           dtype={"label": str, "status": str, "source_image": str, "image_sha256": str},
                           keep_default_na=False)
    required = {"source_image", "label", "status", "image_sha256"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Baseline audit must contain: {sorted(required)}")
    if baseline["source_image"].duplicated().any():
        raise ValueError("The baseline audit contains repeated source paths.")
    baseline_status = baseline.set_index("source_image")["status"].to_dict()

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
    current_sources = {path.relative_to(DATA_DIR).as_posix() for _, path in image_entries}
    if current_sources != set(baseline["source_image"]):
        raise ValueError("Dataset paths differ from the baseline. Restore the same input dataset before comparing.")

    print(f"Dataset: {DATA_DIR}")
    print(f"Classes: {len(EXPECTED_LABELS)}")
    print(f"Image files to process: {len(image_entries)}")
    print("Accepting exactly one detected hand per image.")
    print(f"Black padding on every image: {IMAGE_PADDING_FRACTION:.0%} of width/height per side.")

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
    landmark_rows = []  # Small coordinate arrays, used for the final quality preview.

    with vision.HandLandmarker.create_from_options(options) as detector:
        for label, image_path in tqdm(image_entries, desc="Extracting padded", unit="image"):
            # One audit record for every image, including unsuccessful images.
            record = {column: None for column in AUDIT_COLUMNS}
            record["source_image"] = image_path.relative_to(DATA_DIR).as_posix()
            record["label"] = label
            record["detail"] = ""
            record["review_required"] = False
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

            # Add the SAME border tested in step 05 to EVERY detector input.
            # Hashes above refer to the original pixels, for comparable duplicate checks.
            padded_rgb, pad_x, pad_y = add_image_padding(image_rgb)
            padded_height, padded_width = padded_rgb.shape[:2]
            record["padded_width"], record["padded_height"] = padded_width, padded_height
            record["pad_x"], record["pad_y"] = pad_x, pad_y

            # 4. Extract landmarks. Unexpected detector errors stop execution;
            # they are not silently converted into ordinary bad-image records.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_rgb)
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
                    points_xy, image_width=padded_width, image_height=padded_height
                )
                if features.shape != (42,) or not np.isfinite(features).all():
                    raise ValueError("Expected 42 finite normalized features.")
            except ValueError as error:
                record["status"] = "invalid_landmarks"
                record["detail"] = str(error)
                continue

            record["feature_row"] = len(feature_rows)  # Zero-based data row.
            record["status"] = "ok"
            # A former two-hand image can yield one detection after padding.
            # That does not identify which hand carries the folder's label.
            record["review_required"] = baseline_status[record["source_image"]] == "multiple_hands"
            if result.handedness and result.handedness[0]:
                record["handedness"] = result.handedness[0][0].category_name
            feature_rows.append(features.tolist() + [label])
            landmark_rows.append(points_xy)

    # 6. Build and validate the four output tables after the complete pass.
    features_table, audit_table, summary = build_output_tables(
        feature_rows, audit_rows, EXPECTED_LABELS
    )
    comparison, summary = compare_with_baseline(audit_table, baseline, summary)
    failures = audit_table.loc[audit_table["status"] != "ok"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_tables = [
        (features_table, PROCESSED_DIR / "features_padded.csv"),
        (audit_table, REPORT_DIR / "image_audit_padded.csv"),
        (summary, REPORT_DIR / "extraction_summary_padded.csv"),
        (failures, REPORT_DIR / "failed_images_padded.csv"),
        (comparison, REPORT_DIR / "padding_comparison.csv"),
    ]
    for table, path in output_tables:
        table.to_csv(path, index=False, encoding="utf-8-sig")

    print("\nPer-class extraction results:")
    shown = ["label", "total_images", "baseline_accepted", "accepted", "gained", "lost",
             "success_rate_pct", "review_required"]
    print(summary[shown].to_string(index=False))
    print(f"\nOriginal image files: {len(image_entries)}")
    print(f"Accepted images: {len(features_table)}")
    print(f"Skipped images: {len(failures)}")
    print(f"Feature CSV shape: {features_table.shape}")
    print(f"Overall extraction success: {100 * len(features_table) / len(image_entries):.2f}%")
    print("Extraction success measures usable rows, not classification accuracy.")
    print(f"\nBaseline accepted: {int(summary['baseline_accepted'].sum())}")
    print(f"Newly accepted (gained): {int(summary['gained'].sum())}")
    print(f"Previously accepted but now skipped (lost): {int(summary['lost'].sum())}")
    print(f"Net change: {int(summary['gained'].sum() - summary['lost'].sum()):+d}")
    print("\nPadded extraction statuses:")
    print(audit_table["status"].value_counts().to_string())
    review_count = int(audit_table["review_required"].sum())
    print(f"\nAccepted rows needing hand-selection review: {review_count}")
    print(f"Accepted rows without that review flag: {len(features_table) - review_count}")
    if review_count:
        print("Exclude review_required rows from training until the intended hand is verified.")

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

    metadata_path = PROCESSED_DIR / "features_padded_metadata.json"
    metadata = {
        "feature_file": "features_padded.csv",
        "audit_file": "outputs/image_audit_padded.csv",
        "feature_columns": features_table.columns[:-1].tolist(),
        "label_column": "label",
        "image_padding_fraction_per_side": IMAGE_PADDING_FRACTION,
        "image_padding_color_rgb": [0, 0, 0],
        "running_mode": "IMAGE",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.5,
        "min_hand_presence_confidence": 0.5,
        "normalization": "XY to padded-input pixels; subtract wrist 0; divide by distance 0 to 9",
        "feature_dtype": "float32",
        "requires_review_filter_before_training": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    preview_path = save_quality_preview(comparison, landmark_rows)

    print("\nSaved files:")
    for _, path in output_tables:
        print(path)
    print(metadata_path)
    if preview_path is not None:
        print(preview_path)


if __name__ == "__main__":
    main()

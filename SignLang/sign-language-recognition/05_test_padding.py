"""Compare original inputs with black padding on the 18 selected review images.

Place this file beside preprocessing.py in the project root. Run step 04 first.
Both conditions use the same detector settings as step 03. Results go to
outputs/padding_test; the dataset and features.csv are not modified.
One detected hand with valid coordinates is only a candidate for visual review.
This selected diagnostic sample is not a validation or test set for a classifier.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import normalize_landmarks


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
SELECTION_PATH = PROJECT_DIR / "outputs" / "extraction_review" / "selected_images.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "padding_test"
PADDING_FRACTION = 0.25


def add_padding(image_rgb):
    """Add 25% of width/height on EACH corresponding side; preserve input pixels."""
    height, width = image_rgb.shape[:2]
    pad_x = max(1, round(width * PADDING_FRACTION))
    pad_y = max(1, round(height * PADDING_FRACTION))
    padded = np.pad(
        image_rgb,
        ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded, pad_x, pad_y


def read_rgb_image(path):
    """Use the same decoder and color order as the extraction script."""
    import cv2

    image_bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def inspect_image(detector, image_rgb):
    """Run inference and apply exactly the same acceptance checks as step 03."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    hands = [np.array([[point.x, point.y] for point in hand], dtype=np.float64)
             for hand in result.hand_landmarks]
    detail = ""
    if len(hands) == 0:
        status = "no_hand"
    elif len(hands) > 1:
        status = "multiple_hands"
    else:
        height, width = image_rgb.shape[:2]
        try:
            # Coordinates refer to THIS input's dimensions, including padding.
            features = normalize_landmarks(hands[0], image_width=width, image_height=height)
            if features.shape != (42,) or not np.isfinite(features).all():
                raise ValueError("Expected 42 finite features.")
            status = "ok"
        except ValueError as error:
            status = "invalid_landmarks"
            detail = str(error)
    return {"hands": hands, "status": status, "detail": detail}


def draw_result(ax, image_rgb, result, connections, panel_id, variant):
    """Draw model estimates on the actual input for that condition."""
    height, width = image_rgb.shape[:2]
    ax.imshow(image_rgb)
    for hand_index, points_xy in enumerate(result["hands"]):
        if points_xy.shape != (21, 2) or not np.isfinite(points_xy).all():
            continue
        pixels = points_xy * np.array([width, height])
        color = ["#00e5ff", "#ff9d2e"][hand_index % 2]
        for start, end in connections:
            ax.plot(pixels[[start, end], 0], pixels[[start, end], 1],
                    color=color, linewidth=1.5, zorder=2)
        ax.scatter(pixels[:, 0], pixels[:, 1], s=13, color=color,
                   edgecolors="black", linewidths=0.4, zorder=3)
        # Mark the wrist and fingertips so folded fingers can be inspected.
        for point_id in [0, 4, 8, 12, 16, 20]:
            label = f"H{hand_index + 1}:0" if point_id == 0 else str(point_id)
            ax.annotate(label, pixels[point_id], xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="black",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.4})
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    title_color = "#14715b" if result["status"] == "ok" else "#ae2c2c"
    ax.set_title(f"{panel_id:02d} | {variant} | {result['status']} | hands: {len(result['hands'])}",
                 fontsize=11, color=title_color, pad=8)
    ax.text(0.5, -0.02, f"{width} x {height} px", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)


def print_summary(report):
    """Compare the two new runs, and flag disagreement with the old audit."""
    shown = ["panel_id", "label", "audit_status", "original_status", "padded_status"]
    print("\nPer-image results:")
    print(report[shown].to_string(index=False))
    original_ok = report["original_status"].eq("ok")
    padded_ok = report["padded_status"].eq("ok")
    original_no_hand = report["original_status"].eq("no_hand")
    candidates = original_no_hand & padded_ok
    lost = original_ok & ~padded_ok
    mismatches = report["original_status"].ne(report["audit_status"])
    print(f"\nImages compared: {len(report)}")
    print(f"Original inputs passing the single-hand checks: {int(original_ok.sum())}")
    print(f"Padded inputs passing the single-hand checks: {int(padded_ok.sum())}")
    print(f"New candidates from no_hand: {int(candidates.sum())} / {int(original_no_hand.sum())}")
    print(f"Originally accepted images lost with padding: {int(lost.sum())} / {int(original_ok.sum())}")
    print(f"Original rerun differs from saved audit: {int(mismatches.sum())}")
    if mismatches.any():
        print("Review the audit differences before interpreting the experiment.")
    print("Candidates need visual inspection of landmark placement before acceptance.")
    print("This small selected sample does not measure full-dataset coverage or classification accuracy.")


def main():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(f"Run 04_review_extraction.py first. Missing: {SELECTION_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL_PATH}")
    selected = pd.read_csv(SELECTION_PATH, dtype={"label": str, "status": str, "source_image": str},
                           keep_default_na=False)
    required = {"panel_id", "label", "status", "source_image"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Selection CSV must contain: {sorted(required)}")
    selected = selected.loc[selected["status"].isin(["ok", "no_hand"])].sort_values("panel_id")
    if selected.empty:
        raise ValueError("No accepted or no_hand examples in the selection file.")
    if selected["source_image"].duplicated().any():
        raise ValueError("The selection contains repeated source-image paths.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selected images: {len(selected)}")
    print(f"Inferences to run: {2 * len(selected)} (original + padded)")
    print("Black padding per side: 25% of image width/height.")
    print("Same settings for both inputs: IMAGE, num_hands=2, detection=0.5, presence=0.5.")
    print("An ok result means the single-hand checks passed; inspect the plotted landmarks.")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    connections = [(edge.start, edge.end) for edge in HandLandmarksConnections.HAND_CONNECTIONS]
    records = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for label, group in selected.groupby("label", sort=False):
            fig, axes = plt.subplots(len(group), 2, figsize=(10, 4.2 * len(group) + 1), squeeze=False)
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.91,
                                hspace=0.25, wspace=0.10)
            fig.suptitle(f"Padding experiment | label: {label}", fontsize=17, y=0.98)
            fig.text(0.5, 0.945, "Left: original input | Right: black padding, 25% per side",
                     ha="center", fontsize=11)
            fig.text(0.5, 0.025,
                     "Landmarks are model estimates. IDs: 0 wrist; 4, 8, 12, 16, 20 fingertips.\n"
                     "Same detector settings in both columns. A higher detection count alone is not proof of quality.",
                     ha="center", fontsize=9)
            for row_index, sample in enumerate(group.itertuples(index=False)):
                image_rgb = read_rgb_image(DATA_DIR / sample.source_image)
                padded_rgb, pad_x, pad_y = add_padding(image_rgb)
                original = inspect_image(detector, image_rgb)
                padded = inspect_image(detector, padded_rgb)
                draw_result(axes[row_index, 0], image_rgb, original, connections,
                            sample.panel_id, "original")
                draw_result(axes[row_index, 1], padded_rgb, padded, connections,
                            sample.panel_id, "padded")
                records.append({
                    "panel_id": sample.panel_id,
                    "label": label,
                    "audit_status": sample.status,
                    "original_status": original["status"],
                    "padded_status": padded["status"],
                    "original_hands": len(original["hands"]),
                    "padded_hands": len(padded["hands"]),
                    "original_width": image_rgb.shape[1],
                    "original_height": image_rgb.shape[0],
                    "padded_width": padded_rgb.shape[1],
                    "padded_height": padded_rgb.shape[0],
                    "pad_x_each_side": pad_x,
                    "pad_y_each_side": pad_y,
                    "original_detail": original["detail"],
                    "padded_detail": padded["detail"],
                    "source_image": sample.source_image,
                })
                print(f"{sample.panel_id:02d} | {label} | {original['status']} -> {padded['status']}",
                      flush=True)
            preview_path = OUTPUT_DIR / f"padding_{label}.png"
            fig.savefig(preview_path, dpi=180, facecolor="white")
            plt.close(fig)
            print(f"Saved: {preview_path}", flush=True)

    report = pd.DataFrame(records).sort_values("panel_id")
    report_path = OUTPUT_DIR / "padding_results.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print_summary(report)
    print(f"\nSaved report: {report_path}")
    print(f"Open the comparison PNG files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

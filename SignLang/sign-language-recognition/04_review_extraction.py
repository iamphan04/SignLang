"""Compare accepted images with failures recorded by 03_extract_features.py.

Place this file in the project root, beside 03_extract_features.py.
It reads the existing audit and a small sample of original images. It does not
run MediaPipe, train a classifier, or change the images or extracted features.
The status shown above each image is from the previous extraction run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Gesture Image Data"
AUDIT_PATH = PROJECT_DIR / "outputs" / "image_audit.csv"
REVIEW_DIR = PROJECT_DIR / "outputs" / "extraction_review"
RANDOM_SEED = 42


def select_examples(audit):
    """Select six low-coverage classes and record every preview's source."""
    required = {"label", "status", "source_image"}
    if not required.issubset(audit.columns):
        raise ValueError(f"The audit must include these columns: {sorted(required)}")
    if audit.empty:
        raise ValueError("The image audit is empty.")

    # Count extraction outcomes for each class using the same audit file.
    counts = []
    for label, rows in audit.groupby("label", sort=True):
        accepted = int(rows["status"].eq("ok").sum())
        counts.append({
            "label": label,
            "total_images": len(rows),
            "accepted": accepted,
            "no_hand": int(rows["status"].eq("no_hand").sum()),
            "success_rate_pct": 100 * accepted / len(rows),
        })
    ranking = pd.DataFrame(counts).sort_values(["success_rate_pct", "label"])
    focus = ranking.loc[ranking["no_hand"] > 0].head(6).reset_index(drop=True)

    # Each row compares one accepted image and two failures of the SAME label.
    # Sampling without replacement avoids showing the same file twice.
    # Sorting and a fixed seed make the selection repeatable for the same audit.
    selected = []
    for class_index, label in enumerate(focus["label"]):
        class_rows = audit.loc[audit["label"] == label]
        for status, sample_size, first_column in [("ok", 1, 0), ("no_hand", 2, 1)]:
            pool = class_rows.loc[class_rows["status"] == status]
            pool = pool.sort_values("source_image")
            samples = pool.sample(n=min(sample_size, len(pool)), random_state=RANDOM_SEED)
            for offset, source in enumerate(samples.itertuples(index=False)):
                selected.append({
                    "preview_file": f"review_{class_index // 3 + 1:02d}.png",
                    "panel_id": len(selected) + 1,
                    "row": class_index % 3,
                    "column": first_column + offset,
                    "label": label,
                    "status": status,
                    "source_image": source.source_image,
                })

    # Inspect the less common case separately: multiple detected hands.
    pool = audit.loc[audit["status"] == "multiple_hands"].sort_values("source_image")
    samples = pool.sample(n=min(6, len(pool)), random_state=RANDOM_SEED)
    for index, source in enumerate(samples.itertuples(index=False)):
        selected.append({
            "preview_file": "review_multiple_hands.png",
            "panel_id": len(selected) + 1,
            "row": index // 3,
            "column": index % 3,
            "label": source.label,
            "status": source.status,
            "source_image": source.source_image,
        })

    columns = ["preview_file", "panel_id", "row", "column", "label", "status", "source_image"]
    return focus, pd.DataFrame(selected, columns=columns)


def read_rgb_image(path):
    """Decode images exactly as in extraction, including Windows Unicode paths."""
    import cv2

    image_bytes = path.read_bytes()
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode this image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_preview(page, destination):
    """Show whole images without cropping, changing their proportions, or overlays."""
    is_multiple = destination.name == "review_multiple_hands.png"
    row_count = int(page["row"].max()) + 1
    fig, axes = plt.subplots(row_count, 3, figsize=(12, 4.6 * row_count + 1), squeeze=False)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.055, top=0.91,
                        hspace=0.25, wspace=0.08)

    if is_multiple:
        title = "Images with multiple detected hands"
        subtitle = "Original images | check whether more than one real hand is visible"
    else:
        labels = ", ".join(page.drop_duplicates("row")["label"].tolist())
        title = f"Extraction review | labels: {labels}"
        subtitle = "Each row: same label | accepted image | no_hand example 1 | no_hand example 2"
    fig.suptitle(title, fontsize=17, y=0.98)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10)
    fig.text(0.5, 0.019,
             "Labels and statuses are from image_audit.csv. Panel IDs map to selected_images.csv.",
             ha="center", fontsize=10)

    used_cells = set()
    colors = {"ok": "#13705b", "no_hand": "#b32b2b", "multiple_hands": "#925200"}
    display_status = {"ok": "ACCEPTED", "no_hand": "NO HAND DETECTED",
                      "multiple_hands": "MULTIPLE DETECTED"}
    for record in page.itertuples(index=False):
        ax = axes[record.row, record.column]
        used_cells.add((record.row, record.column))
        image_rgb = read_rgb_image(DATA_DIR / record.source_image)
        height, width = image_rgb.shape[:2]
        ax.imshow(image_rgb)
        ax.set_title(f"{record.panel_id:02d} | {record.label} | {display_status[record.status]}",
                     fontsize=11, color=colors[record.status], pad=9)
        ax.text(0.5, -0.02, f"{width} x {height} px", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)

    for row in range(row_count):
        for column in range(3):
            ax = axes[row, column]
            ax.axis("off")
            if (row, column) not in used_cells:
                ax.text(0.5, 0.5, "No example available", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Run 03_extract_features.py first. Missing: {AUDIT_PATH}")
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    # Labels such as '0' must remain text, just like 'A' or 'B'.
    audit = pd.read_csv(AUDIT_PATH, dtype={"label": str, "status": str, "source_image": str},
                        keep_default_na=False)
    focus, selection = select_examples(audit)
    print("Classes selected for accepted-versus-no_hand review:")
    print(focus.round({"success_rate_pct": 2}).to_string(index=False))
    if selection.empty:
        print("No no_hand or multiple_hands examples were found to review.")
        return

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print("\nCreating previews from original images; MediaPipe is not being rerun.")
    for filename, page in selection.groupby("preview_file", sort=True):
        destination = REVIEW_DIR / filename
        save_preview(page, destination)
        print(f"Saved: {destination}")

    manifest_path = REVIEW_DIR / "selected_images.csv"
    selection.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {manifest_path}")
    print(f"\nImages shown: {len(selection)}")
    print("An accepted status does not guarantee that the landmark positions are correct.")
    print("These targeted samples help investigation; they do not establish a cause for every failure.")
    print(f"Open the PNG files in: {REVIEW_DIR}")


if __name__ == "__main__":
    main()

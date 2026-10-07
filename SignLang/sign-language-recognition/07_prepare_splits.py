r"""Clean the padded feature table and freeze a grouped train/validation/test split.

Run from the project terminal:
    .\.venv\Scripts\python.exe .\07_prepare_splits.py

This step uses the two existing CSV files only. It does not run MediaPipe,
fit a model, change labels, or delete original images.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd


LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]
SPLITS = ("train", "validation", "test")

# Dataset-specific policy, chosen from image counts BEFORE model training.
# All gestures and lighting conditions of a hand ID stay together.
# Other naming families are conservative source proxies, not verified people.
# Ratios are approximate because an entire group must stay in one split.
# handX is consistent with the volunteer-number convention documented in:
# https://mro.massey.ac.nz/server/api/core/bitstreams/09187662-5ebe-4563-8515-3d7e5e1d2a33/content
GROUP_TO_SPLIT = {
    "hand1": "train",
    "hand2": "train",
    "hand3": "validation",
    "hand4": "validation",
    "hand5": "test",
    "photo_letter_parentheses": "train",  # A (1).jpg, B (1).jpg, ...
    "photo_letter_underscore": "validation",  # A_1.jpg, B_1.jpg, ...
    "photo_uuid": "train",
    "photo_IMG_date_underscore": "train",  # IMG_20210605_174833.jpg
    "photo_date": "train",  # 20210607_165156.jpg
    "photo_IMG_date": "test",  # IMG20210607183140.jpg
    "photo_IMG_counter": "test",  # IMG_0209.JPG
}


def require(condition, message):
    """Stop on inconsistent inputs instead of silently misaligning rows."""
    if not condition:
        raise ValueError(message)


def load_and_validate(feature_path, audit_path):
    """Link each feature row to exactly one accepted source image."""
    features = pd.read_csv(feature_path, dtype={"label": str}, keep_default_na=False)
    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    require(
        features.columns.tolist() == FEATURE_COLUMNS + ["label"],
        "Expected x0,y0,...,x20,y20,label in exactly that order.",
    )
    required = {
        "source_image", "label", "status", "feature_row", "image_sha256",
        "detected_hands", "review_required",
    }
    require(required.issubset(audit.columns), f"Audit is missing: {required - set(audit.columns)}")
    require(not audit.source_image.duplicated().any(), "Repeated source paths in audit.")
    require(set(audit.label) == set(LABELS), "Expected all 36 class labels in audit.")
    require(set(features.label) == set(LABELS), "Expected all 36 class labels in features.")

    for source, label in zip(audit.source_image, audit.label):
        parts = PurePosixPath(source).parts
        require(len(parts) >= 2 and parts[0] == label, f"Folder/label mismatch: {source}")

    review = audit.review_required.str.lower().str.strip()
    require(review.isin(["true", "false"]).all(), "Unrecognized review_required flag.")
    audit["review_required"] = review.eq("true")
    accepted = audit.loc[audit.status.eq("ok")].copy()
    row_numbers = pd.to_numeric(accepted.feature_row, errors="raise").to_numpy(dtype=float)
    require(np.isfinite(row_numbers).all(), "Missing feature_row for an accepted image.")
    require((row_numbers == np.floor(row_numbers)).all(), "feature_row must be an integer.")
    accepted["feature_row"] = row_numbers.astype(np.int64)
    accepted = accepted.sort_values("feature_row").reset_index(drop=True)
    require(
        np.array_equal(accepted.feature_row.to_numpy(), np.arange(len(features))),
        "Accepted audit rows must map exactly to feature rows 0 through N-1.",
    )
    require(accepted.label.equals(features.label), "Audit and feature labels do not match.")
    require(
        pd.to_numeric(accepted.detected_hands, errors="raise").eq(1).all(),
        "An accepted row did not have exactly one detected hand.",
    )
    require(
        accepted.image_sha256.str.fullmatch(r"[0-9a-f]{64}").all(),
        "An accepted row is missing its original-image hash.",
    )
    conflicts = accepted.groupby("image_sha256").label.nunique().gt(1)
    require(not conflicts.any(), "Identical original images have conflicting labels; inspect them first.")

    xy = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    require(np.isfinite(xy).all(), "Features contain a non-finite value.")
    require(np.allclose(xy[:, :2], 0.0, atol=1e-6, rtol=0), "Wrist coordinates should be (0,0).")
    palm_lengths = np.linalg.norm(xy[:, 18:20], axis=1)  # Landmark 9, relative to wrist 0.
    require(np.allclose(palm_lengths, 1.0, atol=1e-5, rtol=0), "Reference palm length should be 1.")
    features[FEATURE_COLUMNS] = xy
    return features, audit, accepted


def clean_rows(accepted):
    """Exclude unresolved hand selection, then keep one row per exact image."""
    review_rows = accepted.loc[accepted.review_required].copy()
    review_rows["exclusion_reason"] = "hand_selection_needs_review"
    review_rows["duplicate_of"] = ""
    candidates = accepted.loc[~accepted.review_required].copy()

    # image_sha256 was calculated from ORIGINAL decoded pixels in step 06.
    # Same filename is not enough; different filenames may contain identical pixels.
    duplicated = candidates.duplicated("image_sha256", keep="first")
    duplicate_rows = candidates.loc[duplicated].copy()
    first_source = candidates.drop_duplicates("image_sha256").set_index("image_sha256").source_image
    duplicate_rows["exclusion_reason"] = "exact_image_duplicate"
    duplicate_rows["duplicate_of"] = duplicate_rows.image_sha256.map(first_source)

    clean = candidates.loc[~duplicated].copy().reset_index(drop=True)
    clean["clean_row"] = np.arange(len(clean))
    excluded = pd.concat([review_rows, duplicate_rows], ignore_index=True)
    excluded = excluded.sort_values("feature_row").reset_index(drop=True)
    require(set(clean.label) == set(LABELS), "Cleaning removed an entire class.")
    return clean, excluded


def source_group(source, label):
    """Derive a conservative group from this dataset's filename convention.

    Keep the group independent of the gesture label. For example, all hand1
    letters AND digits share one group. Unknown naming patterns stop the run.
    """
    stem = PurePosixPath(source).stem
    hand = re.fullmatch(r"(hand[1-5])_([a-z0-9])_(?:bot|top|left|right|dif|diff)_seg_\d+_cropped", stem, re.I)
    if hand:
        require(hand[2].upper() == label, f"Gesture code and folder label disagree: {source}")
        return hand[1].lower()

    for pattern, group in (
        (r"([a-z]) \(\d+\)", "photo_letter_parentheses"),
        (r"([a-z])_\d+", "photo_letter_underscore"),
    ):
        match = re.fullmatch(pattern, stem, re.I)
        if match:
            require(match[1].upper() == label, f"Filename and folder label disagree: {source}")
            return group

    for pattern, group in (
        (r"[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12}", "photo_uuid"),
        (r"IMG_\d{8}_\d{6}", "photo_IMG_date_underscore"),
        (r"\d{8}_\d{6}", "photo_date"),
        (r"IMG\d{14}(?:_\d+)?", "photo_IMG_date"),
        (r"IMG_\d{4}(?:\[\d+\]|\s*\(\d+\))?", "photo_IMG_counter"),
    ):
        if re.fullmatch(pattern, stem, re.I):
            return group
    raise ValueError(f"Unknown filename pattern: {source}. Review its source group before splitting.")


def assign_splits(clean):
    """Apply one fixed group policy; never search for a better model score."""
    manifest = clean.copy()
    manifest["source_group"] = [source_group(p, y) for p, y in zip(clean.source_image, clean.label)]
    manifest["split"] = manifest.source_group.map(GROUP_TO_SPLIT)
    require(manifest.split.notna().all(), "A source group has no assigned split.")
    manifest["split_row"] = manifest.groupby("split", sort=False).cumcount()

    counts = pd.crosstab(manifest.label, manifest.split).reindex(index=LABELS, columns=SPLITS, fill_value=0)
    require((counts.to_numpy() > 0).all(), "Every split must contain all 36 classes.")
    require(manifest.groupby("source_group").split.nunique().eq(1).all(), "Source group crosses splits.")
    require(manifest.groupby("image_sha256").split.nunique().eq(1).all(), "Exact image crosses splits.")
    require(not manifest.image_sha256.duplicated().any(), "An exact duplicate remains after cleaning.")
    counts["total"] = counts.sum(axis=1)
    return manifest, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    feature_path = root / "data" / "processed" / "features_padded.csv"
    audit_path = root / "outputs" / "image_audit_padded.csv"
    for path in (feature_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    features, audit, accepted = load_and_validate(feature_path, audit_path)
    clean, excluded = clean_rows(accepted)
    manifest, counts = assign_splits(clean)

    # Exact feature equality can catch another kind of repeated input. Do not
    # delete different hand poses just because their vectors look similar.
    clean_features = features.iloc[manifest.feature_row].reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(clean_features[FEATURE_COLUMNS], index=False)
    identical_vectors = manifest.assign(vector_hash=hashes.to_numpy()).groupby("vector_hash").split.nunique()
    require(identical_vectors.le(1).all(), "Identical feature vectors cross splits; inspect before training.")

    processed = root / "data" / "processed"
    split_dir = processed / "splits"
    output_dir = root / "outputs"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_features.to_csv(processed / "features_clean.csv", index=False)
    for split in SPLITS:
        rows = manifest.loc[manifest.split.eq(split), "feature_row"]
        features.iloc[rows].to_csv(split_dir / f"{split}.csv", index=False)

    manifest_columns = [
        "source_image", "label", "feature_row", "clean_row", "source_group",
        "split", "split_row", "image_sha256",
    ]
    manifest[manifest_columns].to_csv(output_dir / "split_manifest.csv", index=False)
    excluded[["source_image", "label", "feature_row", "exclusion_reason", "duplicate_of"]].to_csv(
        output_dir / "excluded_feature_rows.csv", index=False,
    )
    counts.to_csv(output_dir / "split_summary.csv", index_label="label")
    group_summary = manifest.groupby(["split", "source_group"]).agg(
        images=("source_image", "size"), classes=("label", "nunique"),
    ).reset_index()
    group_summary.to_csv(output_dir / "split_groups.csv", index=False)

    split_sizes = {split: int(manifest.split.eq(split).sum()) for split in SPLITS}
    metadata = {
        "policy_version": "filename_groups_v1",
        "input_file_sha256": {
            "features_padded.csv": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "image_audit_padded.csv": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        },
        "original_images": len(audit),
        "accepted_images": len(features),
        "excluded_review_rows": int(accepted.review_required.sum()),
        "excluded_exact_duplicates": int(excluded.exclusion_reason.eq("exact_image_duplicate").sum()),
        "clean_rows": len(manifest),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": "label",
        "labels": LABELS,
        "group_to_split": GROUP_TO_SPLIT,
        "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_sizes": split_sizes,
        "selection_rule": "Fixed assignment based on source grouping and counts, before fitting any classifier.",
        "limitations": [
            "Photo filename families are source proxies; verified signer IDs are not available for all images.",
            "No overlap of the defined groups does not prove complete signer independence.",
            "Image hashes detect exact decoded-image duplicates, not every near-duplicate.",
            "The split covers accepted single-hand features; report extraction coverage separately from classifier accuracy.",
            "Exploratory preprocessing used dataset samples before this split; a new independent capture is needed for an untouched external evaluation.",
        ],
    }
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Accepted padded rows: {len(features)}")
    print(f"Excluded for hand-selection review: {accepted.review_required.sum()}")
    print(f"Removed exact duplicate copies: {metadata['excluded_exact_duplicates']}")
    print(f"Clean rows: {len(manifest)}")
    print(f"Classes: {manifest.label.nunique()}")
    print(f"Source groups: {manifest.source_group.nunique()}")
    print("\nGrouped split:")
    for split, size in split_sizes.items():
        print(f"  {split:10s} {size:5d} rows | {size / len(manifest):6.2%} | {manifest.loc[manifest.split.eq(split), 'label'].nunique()} classes")
    print("\nPer-class split counts:")
    print(counts.to_string())
    print("\nChecks: feature/audit mapping OK; normalization OK; all classes present.")
    print("Defined source-group overlap: 0 | Exact-image overlap: 0 | Exact-feature overlap: 0")
    print("Photo source groups are filename proxies, not verified signer identities.")
    print("Keep this split fixed when comparing models. Use validation for choices; test only after choosing.")
    print(f"\nSaved model-input CSVs: {split_dir}")
    print(f"Saved clean feature table: {processed / 'features_clean.csv'}")
    print(f"Saved provenance, exclusions and summaries: {output_dir}")
    print("No classifier was trained in this step.")


if __name__ == "__main__":
    main()

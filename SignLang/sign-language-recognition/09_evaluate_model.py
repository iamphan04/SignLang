"""Evaluate the already-selected classifier on the fixed test split."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from model_common import LABELS, dataset_hashes, file_hash, load_bundle, load_split, metrics, write_json


def save_matrix(cm, path, normalized=False):
    values = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix" + (" — fraction within each true class" if normalized else " — image counts"))
    if not normalized:
        cutoff = cm.max() / 2
        for row, col in zip(*np.nonzero(cm)):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=7,
                    color="white" if cm[row, col] > cutoff else "#10252e")
    fig.colorbar(grid, ax=ax, shrink=0.78)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().project_root.resolve()
    bundle = load_bundle(root)
    if bundle["dataset_hashes"] != dataset_hashes(root):
        raise ValueError("The split files changed after training. Restore the matching data/model.")
    x_test, y_test, mapping = load_split(root, "test")
    model = bundle["model"]
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    scores = metrics(y_test, predicted)
    output = root / "outputs" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, predicted, labels=LABELS)
    report = classification_report(y_test, predicted, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T.to_csv(
        output / "classification_report.csv", index_label="label",
    )
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output / "confusion_matrix.csv", index_label="true_label")
    save_matrix(cm, output / "confusion_matrix.png")
    save_matrix(cm, output / "confusion_matrix_normalized.png", normalized=True)

    details = mapping[["source_image", "label", "feature_row", "source_group"]].rename(columns={"label": "true_label"}).copy()
    details["predicted_label"] = predicted
    details["model_score"] = probabilities.max(axis=1)
    details["correct"] = predicted == y_test
    details.to_csv(output / "test_predictions.csv", index=False)
    details.loc[~details.correct].sort_values("model_score", ascending=False).to_csv(output / "misclassified_images.csv", index=False)

    pairs = [{"true_label": LABELS[i], "predicted_label": LABELS[j], "count": int(cm[i, j])}
             for i in range(36) for j in range(36) if i != j and cm[i, j] > 0]
    pair_table = pd.DataFrame(pairs, columns=["true_label", "predicted_label", "count"])
    pair_table = pair_table.sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True])
    pair_table.to_csv(output / "confused_pairs.csv", index=False)
    group_rows = []
    for group, rows in details.groupby("source_group"):
        group_rows.append({"source_group": group, "images": len(rows), "accuracy": float(rows.correct.mean())})
    pd.DataFrame(group_rows).to_csv(output / "test_by_source_group.csv", index=False)

    split_metadata = json.loads((root / "data/processed/splits/split_metadata.json").read_text(encoding="utf-8"))
    result = {"model": bundle["model_name"], "test_images": len(y_test), "test_metrics": scores,
              "model_sha256": file_hash(root / "models/sign_classifier.joblib"),
              "test_csv_sha256": file_hash(root / "data/processed/splits/test.csv"),
              "accepted_images": split_metadata["accepted_images"],
              "original_images": split_metadata["original_images"],
              "extraction_coverage": split_metadata["accepted_images"] / split_metadata["original_images"],
              "scope": "Classifier performance on accepted, cleaned static-image features; not webcam accuracy.",
              "split_limitations": split_metadata["limitations"]}
    write_json(output / "test_metrics.json", result)

    # A presentation-ready text summary generated from actual results, without invented claims.
    lines = ["# Classification results", "", f"Selected model: **{bundle['model_name']}**.", "",
             "Selection used validation macro F1. This same model was fitted on training rows only.", "",
             f"Test images: **{len(y_test)}**. Classes: **36**.", "",
             "| Metric | Test value |", "| --- | ---: |"]
    lines += [f"| {key} | {value:.4f} |" for key, value in scores.items()]
    lines += ["", "## Most frequent directional confusions", "",
              "| True label | Predicted label | Count |", "| --- | --- | ---: |"]
    lines += [f"| {r.true_label} | {r.predicted_label} | {r.count} |" for r in pair_table.head(10).itertuples()]
    per_class = pd.DataFrame({label: report[label] for label in LABELS}).T
    lines += ["", "## Per-class interpretation", "",
              "Highest observed F1: " + ", ".join(f"{label} ({row['f1-score']:.3f}, n={int(row['support'])})"
                                                 for label, row in per_class.sort_values('f1-score', ascending=False).head(5).iterrows()),
              "", "Lowest observed F1: " + ", ".join(f"{label} ({row['f1-score']:.3f}, n={int(row['support'])})"
                                                   for label, row in per_class.sort_values('f1-score').head(5).iterrows()),
              "", "Small per-class test counts make these estimates variable.", "",
              "## Interpretation limits", "",
              "Extraction coverage and classification accuracy measure different stages. Do not multiply them to claim end-to-end accuracy.",
              "Webcam performance must be checked using new captures. Display smoothing and rejection thresholds are not part of these metrics.",
              "Test results describe this selected model. Further changes based on these errors require a new untouched evaluation set.", ""]
    lines += [f"- {item}" for item in split_metadata["limitations"]]
    (output / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected model: {bundle['model_name']} | Test images: {len(y_test)}")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}" + (f" ({value:.2%})" if key == "accuracy" else ""))
    print("\nMost frequent confusions:")
    print(pair_table.head(10).to_string(index=False))
    print(f"\nReports and plots: {output}")
    print("These are static-image classifier results. Webcam behavior still needs a local test.")


if __name__ == "__main__":
    main()

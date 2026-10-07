"""Train three classical classifiers; choose using validation macro F1 only."""

import argparse
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from model_common import (FEATURE_COLUMNS, LABELS, PREPROCESSING, dataset_hashes,
                          load_split, metrics, preprocessing_hash, versions, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    x_train, y_train, _ = load_split(root, "train")
    x_val, y_val, val_mapping = load_split(root, "validation")
    output = root / "outputs" / "training"
    output.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)

    # Settings are fixed before looking at test results. A small, explainable comparison.
    candidates = {
        "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2,
                                               class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                               class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ),
    }
    print(f"Training: {len(y_train)} rows | Validation: {len(y_val)} rows | Features: 42")
    print("Selection: highest validation macro F1; ties use validation accuracy, then model name.")
    rows = []
    predictions = val_mapping[["source_image", "label", "feature_row", "source_group"]].copy()
    for name, model in candidates.items():
        print(f"\nFitting {name} ...", flush=True)
        started = perf_counter()
        # A non-converged logistic fit should not silently become our final model.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        fit_seconds = perf_counter() - started
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        val_scores = metrics(y_val, val_pred)
        train_accuracy = float(np.mean(train_pred == y_train))
        rows.append({"model": name, "train_accuracy": train_accuracy,
                     **{f"validation_{k}": v for k, v in val_scores.items()},
                     "fit_seconds": fit_seconds})
        predictions[name] = val_pred
        report = classification_report(y_val, val_pred, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T.to_csv(
            output / f"{name}_validation_report.csv", index_label="label",
        )
        print(f"Train accuracy: {train_accuracy:.2%} | Validation accuracy: {val_scores['accuracy']:.2%}"
              f" | Validation macro F1: {val_scores['macro_f1']:.4f}")

    comparison = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"], ascending=[False, False, True],
    ).reset_index(drop=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    winner = comparison.iloc[0]["model"]
    best_model = candidates[winner]
    selection = {"selected_model": winner, "selection_metric": "validation_macro_f1",
                 "validation_metrics": {key: float(comparison.iloc[0][f"validation_{key}"])
                                        for key in metrics(y_val, best_model.predict(x_val))},
                 "fitted_on": "train only", "test_used_for_selection": False,
                 "versions": versions()}
    write_json(output / "selection.json", selection)
    write_json(output / "training_environment.json", versions())

    bundle = {"schema_version": 1, "model": best_model, "model_name": winner,
              "feature_columns": FEATURE_COLUMNS, "labels": LABELS,
              "preprocessing": PREPROCESSING, "preprocessing_sha256": preprocessing_hash(root),
              "dataset_hashes": dataset_hashes(root), "selection": selection}
    joblib.dump(bundle, root / "models" / "sign_classifier.joblib", compress=3)

    # A short actual tree excerpt for learning; omitted deeper branches are marked.
    rules = export_text(candidates["DecisionTree"], feature_names=FEATURE_COLUMNS, max_depth=3)
    (output / "decision_tree_excerpt.txt").write_text(rules, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(comparison))
    ax.bar(positions - 0.18, comparison.validation_accuracy, 0.36, label="Validation accuracy", color="#19798a")
    ax.bar(positions + 0.18, comparison.validation_macro_f1, 0.36, label="Validation macro F1", color="#e39b42")
    ax.set_xticks(positions, comparison.model)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("Model comparison on the fixed validation split")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=170)
    plt.close(fig)
    print("\n" + comparison[["model", "train_accuracy", "validation_accuracy", "validation_macro_f1"]].to_string(index=False))
    print(f"\nSelected: {winner}")
    print(f"Saved: {root / 'models' / 'sign_classifier.joblib'}")
    print("The selected model remains fitted on training rows only. Next: 09_evaluate_model.py")


if __name__ == "__main__":
    main()

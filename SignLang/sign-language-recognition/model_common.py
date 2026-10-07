"""Shared data contracts and evaluation helpers for steps 08-11."""

import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

import sys
import types
class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "dummy.py"
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        class MockCD: pass
        return MockCD
sys.modules['sklearn.linear_model._cd_fast'] = MockModule('sklearn.linear_model._cd_fast')

import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from preprocessing import IMAGE_PADDING_FRACTION


LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]
PREPROCESSING = {
    "padding_fraction_per_side": IMAGE_PADDING_FRACTION,
    "padding_rgb": [0, 0, 0],
    "landmark_count": 21,
    "coordinates": "XY only; detector input pixels",
    "position_reference": 0,
    "scale_reference": [0, 9],
    "rotation_normalization": False,
    "input_mirroring": False,
    "detector_mode": "IMAGE",
    "num_hands": 2,
    "accepted_hand_count": 1,
    "min_hand_detection_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocessing_hash(root):
    # Text mode makes Windows CRLF and Unix LF equivalent.
    text = (Path(root) / "preprocessing.py").read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def versions():
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}


def dataset_hashes(root):
    paths = [f"data/processed/splits/{name}.csv" for name in ("train", "validation", "test")]
    paths += ["outputs/split_manifest.csv", "data/processed/splits/split_metadata.json"]
    # Hashing test bytes records identity; test labels/predictions are not used for model selection.
    return {p: file_hash(Path(root) / p) for p in paths}


def load_split(root, name):
    root = Path(root)
    path = root / "data" / "processed" / "splits" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run 07_prepare_splits.py first. Missing: {path}")
    table = pd.read_csv(path, dtype={"label": str})
    if table.columns.tolist() != FEATURE_COLUMNS + ["label"]:
        raise ValueError(f"Wrong columns/order in {path.name}")
    x = table[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = table.label.to_numpy(dtype=str)
    if not np.isfinite(x).all() or set(y) != set(LABELS):
        raise ValueError(f"Invalid features or missing classes in {path.name}")
    if not np.allclose(x[:, :2], 0, atol=1e-6, rtol=0):
        raise ValueError("Wrist normalization does not match training contract.")
    if not np.allclose(np.linalg.norm(x[:, 18:20], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Palm-scale normalization does not match training contract.")

    manifest = pd.read_csv(root / "outputs" / "split_manifest.csv", dtype={"label": str})
    for column in ("source_group", "image_sha256", "source_image"):
        if manifest.groupby(column).split.nunique().max() != 1:
            raise ValueError(f"A {column} crosses splits. Revisit step 07.")
    mapping = manifest.loc[manifest.split.eq(name)].sort_values("split_row").reset_index(drop=True)
    if not np.array_equal(mapping.split_row, np.arange(len(table))) or not np.array_equal(mapping.label, y):
        raise ValueError(f"Split manifest and {name}.csv do not align.")
    return x, y, mapping


def metrics(y_true, y_pred):
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=average, zero_division=0,
        )
        result.update({f"{average}_precision": float(p), f"{average}_recall": float(r),
                       f"{average}_f1": float(f)})
    return result


def load_bundle(root):
    # Thử load VSL model trước
    vsl_path = Path(root) / "models" / "vsl_classifier.joblib"
    if vsl_path.is_file():
        return joblib.load(vsl_path)

    # Load the model produced locally by step 08.
    path = Path(root) / "models" / "sign_classifier.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Run train_vsl_models.py first. Missing: {vsl_path}")
    bundle = joblib.load(path)
    if bundle.get("schema_version") != 1 or bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Unsupported model feature contract. Re-run step 08.")
    if bundle.get("preprocessing") != PREPROCESSING or bundle.get("preprocessing_sha256") != preprocessing_hash(root):
        raise ValueError("The model and preprocessing.py differ. Restore the matching preprocessing file.")
    if list(bundle["model"].classes_) != LABELS or bundle["model"].n_features_in_ != 42:
        raise ValueError("The saved classifier has unexpected classes or feature count.")
    return bundle

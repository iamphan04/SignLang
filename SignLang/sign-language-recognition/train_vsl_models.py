"""
Huấn luyện mô hình nhận dạng VSL (Vietnamese Sign Language).
Đầu vào: data/vsl_collected/vsl_merged.csv
Đầu ra: models/vsl_classifier.joblib và báo cáo phân tích.

Mô hình: Random Forest và MLP (Neural Network).
Phân chia: 80% train / 20% test theo Group (person_id).
"""

import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Mock DLL chặn của Windows
class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "dummy.py"
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        class MockCD: pass
        return MockCD
sys.modules['sklearn.linear_model._cd_fast'] = MockModule('sklearn.linear_model._cd_fast')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "vsl_collected" / "vsl_merged.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "vsl_classifier.joblib"

FEATURE_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]
VSL_LABELS = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]

def main():
    if not DATA_PATH.is_file():
        print(f"❌ Không tìm thấy data: {DATA_PATH}")
        sys.exit(1)

    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Chỉ lấy các label hợp lệ
    df = df[df['label'].isin(VSL_LABELS)]
    
    X = df[FEATURE_COLS].values
    y = df['label'].values
    groups = df['person_id'].values

    print(f"Tổng số mẫu: {len(X)}")
    print("Phân bố người dùng:")
    print(df['person_id'].value_counts())
    
    # 1. Chia dữ liệu 80/20 tách theo người
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]
    
    print("\n--- Tập Huấn Luyện (80%) ---")
    print(f"Kích thước: {len(X_train)}")
    print(f"Người dùng trong Train: {np.unique(g_train)}")
    
    print("\n--- Tập Kiểm Tra (20%) ---")
    print(f"Kích thước: {len(X_test)}")
    print(f"Người dùng trong Test: {np.unique(g_test)}")
    
    # 2. Định nghĩa các mô hình
    # Dùng StandardScaler để chuẩn hóa feature cho MLP (vì MLP nhạy với scale)
    models = {
        "Random Forest": Pipeline([
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1))
        ]),
        "MLP (Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
        ])
    }
    
    best_acc = 0
    best_name = None
    best_model = None
    best_y_pred = None
    
    print("\n--- Bắt đầu huấn luyện ---")
    for name, pipeline in models.items():
        print(f"Đang train {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  → Accuracy trên tập test: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipeline
            best_y_pred = y_pred

    print(f"\n🏆 Mô hình tốt nhất: {best_name} với Accuracy = {best_acc*100:.2f}%")
    
    # 3. Báo cáo chi tiết và Ma trận nhầm lẫn
    print("\n--- Phân tích Lỗi (Error Analysis) ---")
    labels_in_test = np.unique(y_test)
    print("Classification Report:")
    print(classification_report(y_test, best_y_pred, target_names=labels_in_test))
    
    cm = confusion_matrix(y_test, best_y_pred, labels=labels_in_test)
    
    # Tìm các cặp hay bị nhầm nhất
    confused_pairs = []
    for i in range(len(labels_in_test)):
        for j in range(len(labels_in_test)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_label": labels_in_test[i],
                    "pred_label": labels_in_test[j],
                    "count": cm[i, j]
                })
                
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)
    
    if confused_pairs:
        print("🔴 Các cặp ký hiệu hay bị nhầm lẫn nhất (Top 5):")
        for pair in confused_pairs[:5]:
            print(f"  - Thật là '{pair['true_label']}' nhưng đoán nhầm thành '{pair['pred_label']}': {pair['count']} lần")
    else:
        print("✅ Mô hình không đoán nhầm cặp nào trên tập test!")
        
    # Lưu kết quả phân tích vào file
    out_dir = PROJECT_ROOT / "outputs" / "vsl_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_cm = pd.DataFrame(cm, index=labels_in_test, columns=labels_in_test)
    df_cm.to_csv(out_dir / f"confusion_matrix_{best_name.replace(' ', '')}.csv")
    pd.DataFrame(confused_pairs).to_csv(out_dir / f"confused_pairs_{best_name.replace(' ', '')}.csv", index=False)
    
    # 4. Lưu mô hình (Joblib)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURE_COLS,
        "labels": VSL_LABELS,
        "preprocessing": "wrist_normalized",
        "description": "VSL Recognizer",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n💾 Đã lưu mô hình tốt nhất vào: {MODEL_PATH}")


if __name__ == "__main__":
    main()

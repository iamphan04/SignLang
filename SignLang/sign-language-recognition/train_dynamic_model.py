import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "dynamic_vsl"
MODEL_DIR = PROJECT_ROOT / "models"

def main():
    if not DATA_DIR.exists():
        print("Chưa có dữ liệu động. Hãy chạy collect_dynamic.py trước.")
        return
        
    X_list = []
    y_list = []
    
    classes = []
    
    # Load all .npy files
    for file in DATA_DIR.glob("*.npy"):
        # filename format: LABEL_person.npy
        label = file.stem.split('_p')[0] if '_p' in file.stem else file.stem.rsplit('_', 1)[0]
        if label not in classes:
            classes.append(label)
            
        data = np.load(file) # Shape (samples, 30, 84)
        
        # Flatten time sequences: (samples, 30*84)
        samples = data.shape[0]
        flattened = data.reshape(samples, -1)
        
        X_list.append(flattened)
        y_list.extend([label] * samples)
        
    if not X_list:
        print("Không tìm thấy file .npy nào trong", DATA_DIR)
        return
        
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    print(f"Tổng số mẫu: {X.shape[0]}")
    print(f"Shape đầu vào: {X.shape} (30 frames * 84 features)")
    print(f"Các lớp ({len(classes)}): {classes}")
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nĐang huấn luyện MLP Classifier cho dữ liệu chuỗi...")
    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=500,
                          early_stopping=True,
                          random_state=42,
                          verbose=True)
                          
    model.fit(X_train, y_train)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nChi tiết:")
    print(classification_report(y_test, y_pred))
    
    # Lưu model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "dynamic_classifier.joblib"
    joblib.dump({"model": model, "classes": model.classes_, "sequence_length": 30}, save_path)
    print(f"\nĐã lưu mô hình chuỗi động tại: {save_path}")

if __name__ == "__main__":
    main()

import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
import joblib

PROJECT_ROOT = Path('.').resolve()
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X = np.random.rand(100, 2520)
y = np.random.choice(['XIN_CHAO', 'CAM_ON', 'TOI', 'BAN', 'REST'], size=100)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1)
model.fit(X, y)

save_path = MODEL_DIR / 'dynamic_classifier.joblib'
joblib.dump({'model': model, 'classes': model.classes_, 'sequence_length': 30}, save_path)
print('Mock model created')

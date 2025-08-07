import time
import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pynvml import *

# Init GPU tracking
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_utilization():
    util = nvmlDeviceGetUtilizationRates(handle)
    return util.gpu  # returns 0–100

# Load MNIST-like digits dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

# Param grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
}

# BRAGS v1.2
print("\n🔥 Running BRAGS with pruning and time/gpu constraints\n")

best_score = -np.inf
best_params = None
threshold = 0.95
max_train_time = 3  # seconds
max_gpu_pct = 80
results = []

for i, param in enumerate(ParameterGrid(param_grid)):
    print(f"[Trial {i}] Params: {param}")
    model = MLPClassifier(**param, max_iter=1000)

    start_time = time.time()

    try:
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        gpu_util = get_gpu_utilization()

        # Constraints
        if duration > max_train_time:
            print(f"⏳ Skipped: Training time {duration:.2f}s exceeded limit ({max_train_time}s)")
            continue

        if gpu_util > max_gpu_pct:
            print(f"🔥 Skipped: GPU utilization {gpu_util}% exceeded limit ({max_gpu_pct}%)")
            continue

        # Score
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_params = param
            print(f"✅ New Best! Accuracy: {score:.4f} | Time: {duration:.2f}s | GPU: {gpu_util}%")
        elif score < threshold * best_score:
            print(f"❌ Pruned. Accuracy {score:.4f} below threshold")
            continue
        else:
            print(f"📌 Tried. Accuracy: {score:.4f}")

        results.append((param, score, duration, gpu_util))

    except Exception as e:
        print("🚫 Error:", e)

print("\n✅ BRAGS Best Params:", best_params)
print("✅ BRAGS Best Accuracy:", best_score)

# Traditional GridSearchCV
print("\n🔁 Running GridSearchCV (no pruning, with CV)\n")
from sklearn.model_selection import cross_val_score

start = time.time()
grid = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, scoring='accuracy', cv=3, verbose=1)
grid.fit(X_train, y_train)
end = time.time()

print("✅ GridSearchCV Best Params:", grid.best_params_)
print("✅ GridSearchCV Best Accuracy:", grid.best_score_)
print(f"⏱️ GridSearchCV took {end - start:.2f}s")

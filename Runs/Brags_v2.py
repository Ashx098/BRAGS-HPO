import time
import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pynvml import *

# Initialize GPU tracking
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_utilization():
    util = nvmlDeviceGetUtilizationRates(handle)
    return util.gpu  # returns 0–100

# Load MNIST-like digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Parameters
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
}

# BRAGS Config
threshold = 0.95
max_train_time = 5  # seconds
max_gpu_pct = 80
cv = 3

print("\n🔥 Running BRAGS v1.3 with CV =", cv)

best_score = -np.inf
best_params = None
results = []

for i, param in enumerate(ParameterGrid(param_grid)):
    print(f"\n[Trial {i}] Params: {param}")
    model = MLPClassifier(**param, max_iter=1000)

    start_time = time.time()

    try:
        # CV scoring (using CPU internally)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        duration = time.time() - start_time
        mean_score = np.mean(scores)
        gpu_util = get_gpu_utilization()

        # Constraints
        if duration > max_train_time:
            print(f"⏳ Skipped: Took {duration:.2f}s > {max_train_time}s")
            continue

        if gpu_util > max_gpu_pct:
            print(f"🔥 Skipped: GPU usage {gpu_util}% > {max_gpu_pct}%")
            continue

        if mean_score > best_score:
            best_score = mean_score
            best_params = param
            print(f"✅ New Best! Score: {mean_score:.4f} | Time: {duration:.2f}s | GPU: {gpu_util}%")
        elif mean_score < threshold * best_score:
            print(f"❌ Pruned. Score {mean_score:.4f} below {threshold * best_score:.4f}")
            continue
        else:
            print(f"📌 Tried. Score: {mean_score:.4f}")

        results.append({
            "params": param,
            "mean_score": mean_score,
            "scores_per_fold": scores.tolist(),
            "train_time": duration,
            "gpu_pct": gpu_util
        })

    except Exception as e:
        print("🚫 Error during training:", e)
        continue

print("\n✅ BRAGS-CV Best Params:", best_params)
print("✅ BRAGS-CV Best Accuracy:", best_score)

# Final summary
print("\n📊 Trial Summary:")
for r in results:
    print(f"- {r['params']} | Score: {r['mean_score']:.4f} | Time: {r['train_time']:.2f}s | GPU: {r['gpu_pct']}% | CV: {r['scores_per_fold']}")
print("\n🔥 BRAGS v1.3 completed.\n")
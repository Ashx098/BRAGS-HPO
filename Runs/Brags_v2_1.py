import time
import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pynvml import *

# --------------------- GPU Setup --------------------- #
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_utilization():
    util = nvmlDeviceGetUtilizationRates(handle)
    return util.gpu  # 0–100%

# --------------------- Dataset --------------------- #
digits = load_digits()
X, y = digits.data, digits.target

# --------------------- Configs --------------------- #
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
}
threshold = 0.95
max_train_time = 5
max_gpu_pct = 80
cv = 3

# ==================== BRAGS ==================== #
print("\n🔥 Running BRAGS v1.3 with CV =", cv)
brags_start = time.time()

best_score = -np.inf
best_params = None
brags_results = []

for i, param in enumerate(ParameterGrid(param_grid)):
    print(f"\n[Trial {i}] Params: {param}")
    model = MLPClassifier(**param, max_iter=1000)
    start_time = time.time()

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        duration = time.time() - start_time
        mean_score = np.mean(scores)
        gpu_util = get_gpu_utilization()

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

        brags_results.append({
            "params": param,
            "mean_score": mean_score,
            "scores_per_fold": scores.tolist(),
            "train_time": duration,
            "gpu_pct": gpu_util
        })

    except Exception as e:
        print("🚫 Error during training:", e)
        continue

brags_total_time = time.time() - brags_start
print("\n✅ BRAGS-CV Best Params:", best_params)
print("✅ BRAGS-CV Best Accuracy:", best_score)

# BRAGS Summary
print("\n📊 BRAGS Trial Summary:")
for r in brags_results:
    print(f"- {r['params']} | Score: {r['mean_score']:.4f} | Time: {r['train_time']:.2f}s | GPU: {r['gpu_pct']}% | CV: {r['scores_per_fold']}")
print(f"\n⏱️ Total BRAGS Time: {brags_total_time:.2f}s")
print("\n🔥 BRAGS v1.3 completed.\n")

# ==================== GridSearchCV ==================== #
print("🔁 Running GridSearchCV with CV =", cv)
grid_start = time.time()
start_gpu = get_gpu_utilization()

grid = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, scoring='accuracy', cv=cv, verbose=1)
grid.fit(X, y)

end_gpu = get_gpu_utilization()
grid_total_time = time.time() - grid_start

print("\n✅ GridSearchCV Best Params:", grid.best_params_)
print("✅ GridSearchCV Best Accuracy:", grid.best_score_)
print(f"⏱️ Total GridSearch Time: {grid_total_time:.2f}s")
print(f"🔋 GridSearch GPU Before/After: {start_gpu}% ➝ {end_gpu}%")

# RAM usage (optional)
# print(f"💾 RAM Used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

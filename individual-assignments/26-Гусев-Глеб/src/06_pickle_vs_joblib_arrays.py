"""Пример 6. pickle vs joblib на модели с большими numpy-массивами.

KNeighborsClassifier хранит всю обучающую выборку внутри себя
(300 000 x 50 float64 ≈ 115 МБ) — типичный «тяжёлый» объект.
"""
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

D = Path(__file__).parent / "models"
D.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
X = rng.random((300_000, 50))
y = rng.integers(0, 2, 300_000)
model = KNeighborsClassifier().fit(X, y)


def best(fn, n=3):
    res = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        res.append(time.perf_counter() - t0)
    return min(res)


pkl, jl = D / "knn.pkl", D / "knn.joblib"


def pdump():
    with open(pkl, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def pload():
    with open(pkl, "rb") as f:
        pickle.load(f)


print(f"pickle  dump {best(pdump):.3f} с | load {best(pload):.3f} с | "
      f"{os.path.getsize(pkl) >> 20} МБ")
print(f"joblib  dump {best(lambda: joblib.dump(model, jl)):.3f} с | "
      f"load {best(lambda: joblib.load(jl)):.3f} с | {os.path.getsize(jl) >> 20} МБ")
print(f"joblib  load(mmap_mode='r') "
      f"{best(lambda: joblib.load(jl, mmap_mode='r')):.4f} с")
pkl.unlink()
jl.unlink()

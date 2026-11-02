"""Пример 3. Сравнение pickle и joblib с разными методами и уровнями сжатия.

Меряем размер файла, время сохранения и время загрузки. Строим графики.
"""
import pickle
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).parent
TMP = HERE / "models" / "bench"
TMP.mkdir(parents=True, exist_ok=True)
IMAGES = HERE / "output"
IMAGES.mkdir(exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)

REPEATS = 5


def bench(name, dump, load, path):
    dump_t = min(_timed(lambda: dump(path)) for _ in range(REPEATS))
    load_t = min(_timed(lambda: load(path)) for _ in range(REPEATS))
    size = path.stat().st_size / 1024
    print(f"{name:<14} размер {size:8.0f} КБ  dump {dump_t*1000:7.1f} мс  "
          f"load {load_t*1000:7.1f} мс")
    return name, size, dump_t * 1000, load_t * 1000


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def pickle_dump(p):
    with open(p, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


rows = [bench("pickle", pickle_dump, pickle_load, TMP / "m.pkl")]
rows.append(bench("joblib", lambda p: joblib.dump(model, p), joblib.load,
                  TMP / "m.joblib"))
for method, level in [("zlib", 3), ("zlib", 9), ("gzip", 3),
                      ("bz2", 3), ("lzma", 3)]:
    rows.append(bench(
        f"{method}-{level}",
        lambda p, m=method, l=level: joblib.dump(model, p, compress=(m, l)),
        joblib.load,
        TMP / f"m.{method}{level}",
    ))


names = [r[0] for r in rows]
sizes = [r[1] for r in rows]
dumps = [r[2] for r in rows]
loads = [r[3] for r in rows]

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
BASE, ACCENT = "#9AA5B1", "#2B6CB0"
colors = [BASE, BASE] + [ACCENT] * (len(rows) - 2)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
bars = ax.bar(names, sizes, color=colors)
ax.bar_label(bars, labels=[f"{s:.0f}" for s in sizes], padding=3, fontsize=10)
ax.set_ylabel("Размер файла, КБ")
ax.set_title("Размер сохранённой модели (RandomForest, 300 деревьев)")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(IMAGES / "chart_size.png")

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
x = range(len(rows))
w = 0.4
b1 = ax.bar([i - w/2 for i in x], dumps, w, label="сохранение (dump)", color="#2B6CB0")
b2 = ax.bar([i + w/2 for i in x], loads, w, label="загрузка (load)", color="#E07B39")
ax.set_xticks(list(x), names, rotation=20)
ax.set_ylabel("Время, мс")
ax.set_title("Время сохранения и загрузки (минимум из 5 запусков)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMAGES / "chart_time.png")
print("Графики сохранены в", IMAGES)

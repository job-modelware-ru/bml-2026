"""Сравнение CNN с альтернативными подходами на одних и тех же данных.

Модели (все обучаются на 55 000 картинок, проверяются на 10 000 тестовых):

  классическое машинное обучение (scikit-learn), вход — 784 пикселя:
    * логистическая регрессия;
    * случайный лес (200 деревьев);
  «ручные» признаки + линейная модель (как до эпохи CNN):
    * HOG + логистическая регрессия. HOG (гистограммы направлений
      градиентов) — признаки, придуманные человеком: в каждой клетке 4×4
      пикселя считается, какие направления границ там преобладают;
  нейронные сети (PyTorch, 12 эпох, Adam с косинусным уменьшением шага):
    * MLP (784 → 256 → 128 → 10) и MLP-large (784 → 1024 → 512 → 10);
    * TinyCNN — компактная CNN с глобальным усреднением (94 тыс. параметров);
    * SimpleCNN — основная CNN из 04_train_cnn.py.

Нейросети обучаются с тремя seed, в таблице — среднее ± стандартное
отклонение. Обученные модели сохраняются и повторно не обучаются.

Запуск: python 06_compare_models.py            (≈ 45 мин на процессоре)
        python 06_compare_models.py --seeds 1  (только seed 42, ≈ 15 мин)
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (CNN_COLOR, GREY, INK, INK_2, MLP_COLOR, RANDOM_STATE, TEAL,
                    get_device, load_json, save, save_json, section)
from data import as_numpy, get_loaders
from engine import run_experiment
from models import MLP, SimpleCNN, TinyCNN, fmt_int

SEEDS = (42, 1, 2)
EPOCHS = 12

NEURAL = {
    "MLP": (lambda: MLP(), "MLP 784-256-128-10"),
    "MLP_large": (lambda: MLP(hidden=(1024, 512)), "MLP 784-1024-512-10"),
    "TinyCNN": (TinyCNN, "TinyCNN (GAP)"),
    "SimpleCNN": (SimpleCNN, "SimpleCNN"),
}
COLORS = {"LogReg": GREY, "RandomForest": GREY, "HOG+LogReg": TEAL,
          "MLP": MLP_COLOR, "MLP_large": MLP_COLOR, "TinyCNN": CNN_COLOR, "SimpleCNN": CNN_COLOR}


def hog_features(X):
    """HOG-признаки для пачки картинок, X — N×784 в [0, 1]."""
    return np.stack([hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(4, 4),
                         cells_per_block=(2, 2)) for x in X]).astype(np.float32)


def run_sklearn():
    """Обучить три классические модели или взять результат с диска."""
    cached = load_json("compare_sklearn")
    if cached:
        print("  классические модели: результаты загружены с диска")
        return cached
    X_tr, y_tr, _, _, X_te, y_te = as_numpy()
    results = {}

    def evaluate(name, model, Xtr, Xte, extra_time=0.0, n_params=None):
        t0 = time.perf_counter()
        model.fit(Xtr, y_tr)
        fit_time = time.perf_counter() - t0 + extra_time
        acc = float((model.predict(Xte) == y_te).mean())
        results[name] = {"acc": acc, "train_time": fit_time, "n_params": n_params}
        print(f"  {name:<14} acc = {acc:.4f}   обучение {fit_time:6.1f} с")

    evaluate("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
             X_tr, X_te, n_params=784 * 10 + 10)
    evaluate("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                    random_state=RANDOM_STATE), X_tr, X_te)
    t0 = time.perf_counter()
    H_tr, H_te = hog_features(X_tr), hog_features(X_te)
    hog_time = time.perf_counter() - t0
    print(f"  HOG: {H_tr.shape[1]} признаков на картинку, расчёт {hog_time:.0f} с")
    evaluate("HOG+LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
             H_tr, H_te, extra_time=hog_time, n_params=H_tr.shape[1] * 10 + 10)
    save_json(results, "compare_sklearn")
    return results


def run_neural(seeds):
    device = get_device()
    out = {}
    for name, (make, title) in NEURAL.items():
        records = []
        for seed in seeds:
            loaders = get_loaders(seed=seed)
            _, rec = run_experiment(name, make, loaders, device, seed=seed, epochs=EPOCHS)
            records.append(rec)
        accs = np.array([r["test_acc"] for r in records])
        out[name] = {"title": title, "accs": accs.tolist(), "acc": float(accs.mean()),
                     "std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                     "n_params": records[0]["n_params"],
                     "train_time": float(np.mean([r["train_time"] for r in records])),
                     "epoch_time": float(np.mean([np.mean(r["epoch_time"]) for r in records]))}
    return out


def plot_accuracy(table):
    """Горизонтальные столбцы: точность ± std."""
    names = sorted(table, key=lambda n: table[n]["acc"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(names))
    acc = [100 * table[n]["acc"] for n in names]
    err = [100 * table[n].get("std", 0) for n in names]
    ax.barh(y, acc, xerr=err, color=[COLORS[n] for n in names], height=0.62,
            error_kw=dict(ecolor=INK, capsize=3, lw=1))
    for yi, a, n in zip(y, acc, names):
        ax.text(a + 0.25, yi, f"{a:.2f} %", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [table[n]["label"] for n in names], fontsize=10)
    ax.set_xlim(80, 95)
    ax.set_xlabel("точность на тесте, %")
    ax.grid(axis="y", visible=False)
    ax.set_title("Fashion-MNIST: точность на 10 000 тестовых картинок", fontsize=12)
    fig.tight_layout()
    save(fig, "compare_accuracy")


def plot_params(table):
    """Точность против числа параметров (логарифмическая шкала)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n in ("MLP", "MLP_large", "TinyCNN", "SimpleCNN", "LogReg", "HOG+LogReg"):
        r = table[n]
        ax.scatter(r["n_params"], 100 * r["acc"], s=90, color=COLORS[n], zorder=3,
                   edgecolor="white", lw=1.5)
        offset = {"MLP": (8, -16)}.get(n, (8, -4) if "CNN" in n else (8, 6))
        ax.annotate(r["label"], (r["n_params"], 100 * r["acc"]), textcoords="offset points",
                    xytext=offset, fontsize=10, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("число обучаемых параметров (лог. шкала)")
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("Параметры и точность", fontsize=12)
    ax.set_xlim(3e3, 1e7)
    fig.tight_layout()
    save(fig, "compare_params")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help="сколько seed использовать для нейросетей (1–3)")
    seeds = SEEDS[:parser.parse_args().seeds]

    section("1. Классические модели (scikit-learn)")
    sk = run_sklearn()
    section(f"2. Нейронные сети (seed: {', '.join(map(str, seeds))})")
    nn_res = run_neural(seeds)

    labels = {"LogReg": "Логистическая регрессия", "RandomForest": "Случайный лес",
              "HOG+LogReg": "HOG + лог. регрессия", "MLP": "MLP (2 скрытых слоя)",
              "MLP_large": "MLP-large", "TinyCNN": "TinyCNN", "SimpleCNN": "SimpleCNN"}
    table = {}
    for name, r in {**sk, **nn_res}.items():
        table[name] = dict(r, label=labels[name])
    section("3. Итоговая таблица")
    print(f"  {'модель':<26}{'точность, %':>16}{'параметров':>14}{'обучение, с':>13}")
    for name in sorted(table, key=lambda n: table[n]["acc"]):
        r = table[name]
        acc = f"{100 * r['acc']:.2f}" + (f" ± {100 * r['std']:.2f}" if r.get("std") else "")
        params = fmt_int(r["n_params"]) if r.get("n_params") else "—"
        print(f"  {r['label']:<26}{acc:>16}{params:>14}{r['train_time']:>13.0f}")
    print("\n  Время обучения — на процессоре (2 потока), включая расчёт HOG для HOG-модели.")
    plot_accuracy(table)
    plot_params(table)
    save_json(table, "compare_models")


if __name__ == "__main__":
    main()

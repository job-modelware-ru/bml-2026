# -*- coding: utf-8 -*-
"""
Егор, вот тип наша структура проги

Обучение с частичным привлечением учителя 
Демонстрация к презентации команды 27.

Что делаем:
  ЧАСТЬ 1 Численный пошаговый пример самообучения на данных из 6 точек
  ЧАСТЬ 2 Реальная задача: распознавание рукописных цифр
           когда размечено всего 5% объектов. 
           Сравниваем три подхода:
             - обучение только на размеченных данных ;
             - самообучение;
             - распространение меток по графу .
  ЧАСТЬ 3 Зависимость качества от доли размеченных данных .

"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(42)

#цвета графиков
RED = "#FF0008"
DARKRED = "#81191A"
LIGHTRED = "#FC7F7F"
GRAY = "#99999A"
DARKGRAY = "#4A4A4A"



# ЧАСТЬ 1
def numeric_example():
    print("=" * 70)
    print("ЧАСТЬ 1. Пошаговый численный пример самообучения")
    print("=" * 70)
    print(

"""
Задача: бинарная классификация, признак x одномерный.
Размечено учителем всего 2 объекта:
    x = 1   ->  класс 0
    x = 10  ->  класс 1
Неразмеченные объекты:  x = 2, 3, 4, 6, 8, 9.
Классифицируем методом ближайшего центроида.
 объект добавляется в обучающую выборку,
только если разница расстояний до центроидов
не меньше 2 (дельта). сомнительные точки отбрасываются.
"""
    )

    # Шаг 1. Обучение на размеченных данных ---
    c0 = 1.0  # центроид класса 0
    c1 = 10.0  # центроид класса 1
    print("Шаг 1. Центроиды по размеченным данным:")
    print(f"    центроид класса 0: c0 = {c0}")
    print(f"    центроид класса 1: c1 = {c1}")

    # Шаг 2. Предсказание для неразмеченных 
    unlabeled = [2.0, 3.0, 4.0, 6.0, 8.0, 9.0]
    print("\nШаг 2. Расстояния от неразмеченных точек до центроидов:")
    print(f"    {'x':>4} | {'d(x,c0)':>8} | {'d(x,c1)':>8} | "
          f"{'прогноз':>8} | {'отступ':>7} | {'берём?':>7}")
    accepted = {0: [], 1: []}
    for x in unlabeled:
        d0 = abs(x - c0)
        d1 = abs(x - c1)
        pred = 0 if d0 < d1 else 1
        margin = abs(d0 - d1)
        take = margin >= 2.0
        if take:
            accepted[pred].append(x)
        print(f"    {x:>4} | {d0:>8.1f} | {d1:>8.1f} | "
              f"{pred:>8} | {margin:>7.1f} | {'да' if take else 'нет':>7}")

    # Шаг 3. Дообучение на псевдометках 
    print("\nШаг 3. Добавляем уверенные псевдометки и пересчитываем центроиды:")
    new_c0 = np.mean([1.0] + accepted[0])
    new_c1 = np.mean([10.0] + accepted[1])
    print(f"    класс 0: точки {[1.0] + accepted[0]} -> c0' = {new_c0:.2f}")
    print(f"    класс 1: точки {[10.0] + accepted[1]} -> c1' = {new_c1:.2f}")

    #  Шаг 4. Что изменилось 
    b_old = (c0 + c1) / 2
    b_new = (new_c0 + new_c1) / 2
    print("\nШаг 4. Разделяющая граница (середина между центроидами):")
    print(f"    была:     x = {b_old:.2f}")
    print(f"    стала:    x = {b_new:.2f}")
    print(


"""
Вывод, Егор смотри, точка x = 6 отклонилась, как сомнительная так как отступ = 1.0 ,
а уверенные псевдометки сдвинули границу с 5.50 на 5.75 немного ближе
к реальной структуре данных. получается что моделька уточнила решение без нового вызова учителя.
"""
    )


# ЧАСТЬ 2
def digits_demo(n_labeled_ratio=0.05):
    print("=" * 70)
    print("ЧАСТЬ 2. Распознавание цифр , размечено "
          f"{n_labeled_ratio:.0%} объектов")
    print("=" * 70)

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Оставляем метки только у n_labeled_ratio части обучающей выборки
    n_labeled = int(len(X_train) * n_labeled_ratio)
    idx = rng.permutation(len(X_train))
    labeled_idx = idx[:n_labeled]
    y_train_semi = np.full(len(X_train), -1)
    y_train_semi[labeled_idx] = y_train[labeled_idx]
    print(f"\nОбучающая выборка: {len(X_train)} объектов, "
          f"из них размечено: {n_labeled}; тест: {len(X_test)}")

    results = {}

    # только размеченные данные
    base = LogisticRegression(max_iter=2000)
    base.fit(X_train[labeled_idx], y_train[labeled_idx])
    results["Только размеченные (5%)"] = accuracy_score(
        y_test, base.predict(X_test))

    # Самообучение
    self_train = SelfTrainingClassifier(
        LogisticRegression(max_iter=2000), threshold=0.9, verbose=False)
    self_train.fit(X_train, y_train_semi)
    results["Самообучение (порог 0.9)"] = accuracy_score(
        y_test, self_train.predict(X_test))
    n_used = int(np.sum(self_train.labeled_iter_ >= 0))
    print(f"Самообучение: итераций {len(np.unique(self_train.labeled_iter_))}, "
          f"в итоге задействовано объектов: {n_used} из {len(X_train)}")

    # Распространение по графу 
    spread = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=30)
    spread.fit(X_train, y_train_semi)
    results["Распространение меток (kNN-граф)"] = accuracy_score(
        y_test, spread.predict(X_test))

    #  Верхняя граница: все метки известны 
    full = LogisticRegression(max_iter=2000)
    full.fit(X_train, y_train)
    results["Все метки (100%) — ориентир"] = accuracy_score(
        y_test, full.predict(X_test))

    print("\nТочность на тесте:")
    for name, acc in results.items():
        print(f"    {name:<38} {acc:.4f}")

    #  График точности от доли размеченных данных 
    ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    acc_base, acc_self, acc_spread = [], [], []
    for r in ratios:
        n_lab = max(10, int(len(X_train) * r))
        idx_r = idx[:n_lab]
        y_semi = np.full(len(X_train), -1)
        y_semi[idx_r] = y_train[idx_r]

        m_base = LogisticRegression(max_iter=2000).fit(
            X_train[idx_r], y_train[idx_r])
        m_self = SelfTrainingClassifier(
            LogisticRegression(max_iter=2000), threshold=0.9).fit(
            X_train, y_semi)
        m_spread = LabelSpreading(kernel="knn", n_neighbors=7).fit(
            X_train, y_semi)

        acc_base.append(accuracy_score(y_test, m_base.predict(X_test)))
        acc_self.append(accuracy_score(y_test, m_self.predict(X_test)))
        acc_spread.append(accuracy_score(y_test, m_spread.predict(X_test)))

    print("\nТочность в зависимости от доли размеченных данных:")
    print(f"    {'доля':>6} | {'только разм.':>13} | {'самообуч.':>10} | "
          f"{'распростр.':>11}")
    for r, a, b, c in zip(ratios, acc_base, acc_self, acc_spread):
        print(f"    {r:>6.0%} | {a:>13.4f} | {b:>10.4f} | {c:>11.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    x_pct = [int(r * 100) for r in ratios]
    ax.plot(x_pct, acc_base, color=GRAY, marker="o", lw=2,
            label="Только размеченные данные")
    ax.plot(x_pct, acc_self, color=RED, marker="o", lw=2.5,
            label="Самообучение")
    ax.plot(x_pct, acc_spread, color=DARKRED, marker="s", lw=2.5,
            linestyle="--", label="Распространение меток")
    ax.axhline(results["Все метки (100%) — ориентир"], color=DARKGRAY,
               lw=1, linestyle=":", label="Все метки (100%)")
    for xi, yi in zip(x_pct, acc_self):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=RED)
    ax.set_xscale("log")
    ax.set_xticks(x_pct)
    ax.set_xticklabels([f"{p}%" for p in x_pct])
    ax.set_xlabel("Доля размеченных объектов в обучающей выборке")
    ax.set_ylabel("Точность на тесте")
    ax.set_ylim(0.5, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("ssl_curve.png")
    plt.close(fig)

    return results


# ЧАСТЬ 3
def spread_illustration():
    print("=" * 70)
    print("ЧАСТЬ 3. как метки расползаются по «лентам» данных")
    print("=" * 70)
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=200, noise=0.08, random_state=42)
    y_semi = np.full(len(y), -1)
    for cls in (0, 1):
        y_semi[np.where(y == cls)[0][:2]] = cls  # по 2 метки на класс

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=200)
    titles = ["Старт: 4 метки из 200",
              "LabelSpreading: после обучения",
              "Истинные метки"]
    for k in (0, 1):
        lab = y_semi if k == 0 else None
        ax = axes[k]
        if k == 0:
            ax.scatter(X[y_semi == -1, 0], X[y_semi == -1, 1],
                       c="#D1D3D4", s=14, label="без метки")
            for cls, col in ((0, DARKGRAY), (1, RED)):
                m = y_semi == cls
                ax.scatter(X[m, 0], X[m, 1], c=col, s=60,
                           edgecolors="black", linewidths=0.8,
                           label=f"метка класса {cls}")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        else:
            model = LabelSpreading(kernel="knn", n_neighbors=7).fit(X, y_semi)
            y_hat = model.transduction_
            for cls, col in ((0, GRAY), (1, LIGHTRED)):
                m = y_hat == cls
                ax.scatter(X[m, 0], X[m, 1], c=col, s=14)
            for cls, col in ((0, DARKGRAY), (1, RED)):
                m = y_semi == cls
                ax.scatter(X[m, 0], X[m, 1], c=col, s=60,
                           edgecolors="black", linewidths=0.8)
            acc = accuracy_score(y, y_hat)
            ax.set_title(f"{titles[k]}\nточность {acc:.1%}", fontsize=10)
            axes[2].scatter(X[y == 0, 0], X[y == 0, 1], c=GRAY, s=14)
            axes[2].scatter(X[y == 1, 0], X[y == 1, 1], c=LIGHTRED, s=14)
    axes[0].set_title(titles[0], fontsize=10)
    axes[2].set_title(titles[2], fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("ssl_spread.png")
    plt.close(fig)
    print("Файлы ssl_curve.png и ssl_spread.png сохранены.")


if __name__ == "__main__":
    numeric_example()
    digits_demo()
    spread_illustration()

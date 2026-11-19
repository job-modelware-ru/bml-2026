"""Добавление полиномиальных признаков для нелинейной зависимости.

Запуск: python src/solution.py
Результаты (CSV и графики) сохраняются в каталоге results/ рядом с README.md.

Что делает скрипт:
  1. Показывает, какие столбцы создаёт PolynomialFeatures и как быстро растёт их число.
  2. Квадратичная зависимость: линейная регрессия на x против регрессии на [x, x^2].
  3. Подбор степени полинома 10-кратной кросс-валидацией (степени 1..15),
     затем проверка выбранной модели на отложенном тесте.
  4. Регуляризация: модель 15-й степени с StandardScaler + Ridge.
  5. Два признака: член взаимодействия x1*x2.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEGREES = list(range(1, 16))
RIDGE_ALPHAS = [1e-6, 1e-4, 1e-2, 1.0, 100.0]



def poly_model(degree: int, alpha: float | None = None):
    """Конвейер: полиномиальные признаки -> (масштабирование) -> линейная модель.

    include_bias=False: столбец из единиц не нужен, свободный член
    LinearRegression и Ridge обучают сами.
    """
    if alpha is None:
        return make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )

    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def true_fun(x: np.ndarray) -> np.ndarray:
    """Нелинейная «истинная» зависимость (как в примере scikit-learn)."""
    return np.cos(1.5 * np.pi * x)


def cv_mse(model, x: np.ndarray, y: np.ndarray, folds: KFold) -> tuple[float, float]:
    """Среднее и стандартное отклонение MSE по фолдам."""
    scores = -cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=folds)
    return float(scores.mean()), float(scores.std(ddof=1))


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)



def part1_what_it_does() -> None:
    header("1. Что делает PolynomialFeatures")

    x = np.array([[2.0, 3.0]])
    poly = PolynomialFeatures(degree=2)
    print("Исходный объект [a, b] =", x[0])
    print("Новые признаки:", list(poly.fit(x).get_feature_names_out(["a", "b"])))
    print("Значения:      ", poly.transform(x)[0])
    poly_i = PolynomialFeatures(degree=2, interaction_only=True)
    print("interaction_only=True:", list(poly_i.fit(x).get_feature_names_out(["a", "b"])))

    print("\nЧисло признаков (с единичным столбцом) = C(n + d, d):")
    print(f"{'n признаков':>12} | {'d=2':>6} | {'d=3':>6} | {'d=5':>8}")
    for n in (1, 2, 5, 10, 20):
        counts = []
        for d in (2, 3, 5):
            got = PolynomialFeatures(degree=d).fit(np.zeros((1, n))).n_output_features_
            assert got == comb(n + d, d)  # проверяем формулу
            counts.append(got)
        print(f"{n:>12} | {counts[0]:>6} | {counts[1]:>6} | {counts[2]:>8}")



def part2_quadratic(out: Path, seed: int) -> None:
    header("2. Квадратичная зависимость: линейная модель против полиномиальной")

    rng = np.random.RandomState(seed)
    x = rng.uniform(-3, 3, 200)
    y = 0.5 * x**2 - x + 2 + rng.normal(0, 0.5, x.size)
    X = x.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    models = {
        "Линейная регрессия (x)": LinearRegression().fit(X_tr, y_tr),
        "Полиномиальная, степень 2 (x, x^2)": poly_model(2).fit(X_tr, y_tr),
    }
    r2 = {}
    for name, model in models.items():
        pred = model.predict(X_te)
        r2[name] = r2_score(y_te, pred)
        print(f"{name:<38} MSE на тесте = {mean_squared_error(y_te, pred):.3f}   "
              f"R^2 = {r2[name]:.3f}")

    lr = models["Полиномиальная, степень 2 (x, x^2)"].named_steps["linearregression"]
    print(f"Найденные коэффициенты: y = {lr.intercept_:.2f} "
          f"{lr.coef_[0]:+.2f}*x {lr.coef_[1]:+.2f}*x^2   "
          f"(истина: 2.00 -1.00*x +0.50*x^2)")

    grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=12, alpha=0.5, color="#888888", label="данные")
    for (name, model), color, label in zip(models.items(), ["#d1495b", "#00798c"],
                                           ["линейная", "степень 2"]):
        ax.plot(grid, model.predict(grid), color=color, lw=2.5,
                label=f"{label}, R² = {r2[name]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Одна и та же линейная регрессия на разных признаках")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "1_linear_vs_poly.png", dpi=150)
    plt.close(fig)



def part3_degree_selection(out: Path, seed: int):
    header("3. Подбор степени полинома (10-кратная кросс-валидация)")

    rng = np.random.RandomState(seed)
    x = np.sort(rng.rand(30))
    y = true_fun(x) + rng.randn(30) * 0.1
    X = x.reshape(-1, 1)


    rng_test = np.random.RandomState(seed + 1)
    x_test = rng_test.rand(200)
    y_test = true_fun(x_test) + rng_test.randn(200) * 0.1
    X_test = x_test.reshape(-1, 1)

    folds = KFold(n_splits=10, shuffle=True, random_state=seed)
    rows = []
    for d in DEGREES:
        model = poly_model(d).fit(X, y)
        mean, std = cv_mse(poly_model(d), X, y, folds)
        rows.append({
            "degree": d,
            "train_mse": mean_squared_error(y, model.predict(X)),
            "cv_mse_mean": mean,
            "cv_mse_std": std,
            "test_mse": mean_squared_error(y_test, model.predict(X_test)),
        })
    save_csv(rows, out / "degree_cv_results.csv")

    print(f"{'степень':>7} | {'MSE обучение':>12} | {'MSE CV (среднее ± SD)':>22} | {'MSE тест':>9}")
    for r in rows:
        cv_text = f"{r['cv_mse_mean']:.4f} ± {r['cv_mse_std']:.4f}"
        print(f"{r['degree']:>7} | {r['train_mse']:>12.4f} | {cv_text:>22} | {r['test_mse']:>9.4f}")
    best = min(rows, key=lambda r: r["cv_mse_mean"])["degree"]
    print(f"Лучшая степень по кросс-валидации: {best}")
    print(f"Отложенный тест (200 новых точек): степень 1 — {rows[0]['test_mse']:.4f}, "
          f"степень {best} — {rows[best - 1]['test_mse']:.4f}, "
          f"степень 15 — {rows[-1]['test_mse']:.4f}")


    cv_mean = [r["cv_mse_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(DEGREES, [r["train_mse"] for r in rows], "o-", color="#00798c",
            label="обучающая выборка")
    ax.plot(DEGREES, cv_mean, "o-", color="#d1495b", label="кросс-валидация")
    ax.axvline(best, color="#555555", ls="--", lw=1)
    ax.text(best + 0.2, 0.95, f"лучшая: {best}", color="#555555",
            transform=ax.get_xaxis_transform(), va="top")
    ax.set_yscale("log")
    ax.set_xlabel("степень полинома")
    ax.set_ylabel("MSE (лог. шкала)")
    ax.set_title("Ошибка на обучении падает, на новых данных растёт")
    ax.set_xticks(DEGREES)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.1))
    fig.tight_layout()
    fig.savefig(out / "2_error_vs_degree.png", dpi=150)
    plt.close(fig)


    grid = np.linspace(0, 1, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, d, t in zip(axes, [1, best, 15], ["недообучение", "хорошо", "переобучение"]):
        model = poly_model(d).fit(X, y)
        ax.plot(grid, true_fun(grid), color="#888888", ls="--", label="истинная функция")
        ax.plot(grid, model.predict(grid), color="#00798c", lw=2.5, label="модель")
        ax.scatter(x, y, s=18, color="#d1495b", zorder=3, label="данные")
        ax.set_ylim(-2, 2)
        ax.set_title(f"степень {d}: {t}\nMSE CV = {cv_mean[d - 1]:.3g}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_under_over_fitting.png", dpi=150)
    plt.close(fig)

    return X, y, X_test, y_test, folds, rows



def part4_ridge(out: Path, X, y, X_test, y_test, folds, degree_rows) -> None:
    header("4. Регуляризация: степень 15 с Ridge")

    rows = [{
        "model": "LinearRegression",
        "alpha": 0.0,
        "cv_mse_mean": degree_rows[-1]["cv_mse_mean"],
        "cv_mse_std": degree_rows[-1]["cv_mse_std"],
        "test_mse": degree_rows[-1]["test_mse"],
    }]
    for a in RIDGE_ALPHAS:
        mean, std = cv_mse(poly_model(15, alpha=a), X, y, folds)
        test = mean_squared_error(y_test, poly_model(15, alpha=a).fit(X, y).predict(X_test))
        rows.append({"model": "Ridge", "alpha": a, "cv_mse_mean": mean,
                     "cv_mse_std": std, "test_mse": test})
    save_csv(rows, out / "ridge_cv_results.csv")

    for r in rows:
        name = "Без регуляризации (МНК)" if r["model"] == "LinearRegression" else f"Ridge, alpha = {r['alpha']:g}"
        print(f"{name:<26} MSE CV = {r['cv_mse_mean']:.4f} ± {r['cv_mse_std']:.4f}   "
              f"MSE тест = {r['test_mse']:.4f}")
    best = min(rows[1:], key=lambda r: r["cv_mse_mean"])
    print(f"Лучшее alpha по кросс-валидации: {best['alpha']:g}")

    grid = np.linspace(0, 1, 300).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, true_fun(grid), color="#888888", ls="--", label="истинная функция")
    ax.plot(grid, poly_model(15).fit(X, y).predict(grid), color="#d1495b", lw=2,
            label=f"степень 15, МНК (MSE CV = {rows[0]['cv_mse_mean']:.2g})")
    ax.plot(grid, poly_model(15, alpha=best["alpha"]).fit(X, y).predict(grid),
            color="#00798c", lw=2.5,
            label=f"степень 15, Ridge α={best['alpha']:g} (MSE CV = {best['cv_mse_mean']:.2g})")
    ax.scatter(X.ravel(), y, s=18, color="#333333", zorder=3, label="данные")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Штраф на коэффициенты укрощает высокую степень")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "4_ridge.png", dpi=150)
    plt.close(fig)



def part5_interaction(seed: int) -> None:
    header("5. Два признака: зависимость через произведение x1*x2")

    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = 3 * X[:, 0] * X[:, 1] + X[:, 0] + rng.normal(0, 0.3, 300)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    lin = LinearRegression().fit(X_tr, y_tr)
    inter = make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False),
                          LinearRegression()).fit(X_tr, y_tr)
    names = inter.named_steps["polynomialfeatures"].get_feature_names_out(["x1", "x2"])
    coefs = inter.named_steps["linearregression"].coef_
    print(f"Линейная регрессия (x1, x2):             R^2 на тесте = {r2_score(y_te, lin.predict(X_te)):.3f}")
    print(f"С членом взаимодействия (x1, x2, x1*x2): R^2 на тесте = {r2_score(y_te, inter.predict(X_te)):.3f}")
    print("Коэффициенты:", ", ".join(f"{n}: {c:+.2f}" for n, c in zip(names, coefs)),
          "  (истина: x1: +1, x2: 0, x1 x2: +3)")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="random_state (по умолчанию: 0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results",
                        help="куда сохранять CSV и графики (по умолчанию: results/)")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    part1_what_it_does()
    part2_quadratic(out, args.seed)
    X, y, X_test, y_test, folds, degree_rows = part3_degree_selection(out, args.seed)
    part4_ridge(out, X, y, X_test, y_test, folds, degree_rows)
    part5_interaction(args.seed)

    print(f"\nРезультаты: {out.resolve()}")


if __name__ == "__main__":
    main()

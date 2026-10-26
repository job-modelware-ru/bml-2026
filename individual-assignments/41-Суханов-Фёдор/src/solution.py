"""Исследование влияния max_features на случайный лес.

Запуск: python src/solution.py
Результаты сохраняются в каталоге results/ рядом с README.md.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # построение графика без открытого окна
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def mean_tree_correlation(model: RandomForestClassifier, x_valid: np.ndarray) -> float:
    """Средняя корреляция прогнозов разных деревьев на валидационных объектах."""
    predictions = np.asarray([tree.predict(x_valid) for tree in model.estimators_])
    # Константное предсказание имеет нулевую дисперсию: корреляция для него не определена.
    predictions = predictions[np.std(predictions, axis=1) > 0]
    if len(predictions) < 2:
        return float("nan")
    correlations = np.corrcoef(predictions)
    upper_triangle = np.triu_indices(len(predictions), k=1)
    return float(np.mean(correlations[upper_triangle]))


def evaluate_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_features: int,
    folds: StratifiedKFold,
    n_estimators: int,
    seed: int,
) -> dict[str, float | int]:
    """Все кандидаты оцениваются на одинаковых фолдах и при одинаковых настройках."""
    accuracy, auc, train_accuracy, correlation, fit_time = [], [], [], [], []
    for train_idx, valid_idx in folds.split(x_train, y_train):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(x_train[train_idx], y_train[train_idx])
        fit_time.append(time.perf_counter() - start)

        x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
        accuracy.append(accuracy_score(y_valid, model.predict(x_valid)))
        auc.append(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        train_accuracy.append(accuracy_score(y_train[train_idx], model.predict(x_train[train_idx])))
        correlation.append(mean_tree_correlation(model, x_valid))

    return {
        "max_features": max_features,
        "cv_accuracy_mean": float(np.mean(accuracy)),
        "cv_accuracy_std": float(np.std(accuracy, ddof=1)),
        "cv_roc_auc_mean": float(np.mean(auc)),
        "cv_roc_auc_std": float(np.std(auc, ddof=1)),
        "train_accuracy_mean": float(np.mean(train_accuracy)),
        "tree_correlation_mean": float(np.nanmean(correlation)),
        "fit_seconds_mean": float(np.mean(fit_time)),
    }


def save_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    x = np.asarray([row["max_features"] for row in rows], dtype=int)
    auc = np.asarray([row["cv_roc_auc_mean"] for row in rows], dtype=float)
    auc_std = np.asarray([row["cv_roc_auc_std"] for row in rows], dtype=float)
    correlation = np.asarray([row["tree_correlation_mean"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].errorbar(x, auc, yerr=auc_std, marker="o", capsize=3, color="#2364aa")
    axes[0].set_title("Качество классификации")
    axes[0].set_ylabel("ROC-AUC: среднее ± SD по фолдам")
    axes[1].plot(x, correlation, marker="o", color="#d1495b")
    axes[1].set_title("Сходство деревьев")
    axes[1].set_ylabel("Средняя попарная корреляция прогнозов")
    for axis in axes:
        axis.set_xlabel("max_features (из 30 признаков)")
        axis.set_xticks(x)
        axis.axvline(5, linestyle="--", linewidth=1, color="gray", label="sqrt(30) ≈ 5")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=120, help="число деревьев (по умолчанию: 120)")
    parser.add_argument("--folds", type=int, default=5, help="число фолдов CV (по умолчанию: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random_state (по умолчанию: 42)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    if args.n_estimators < 2 or args.folds < 2:
        parser.error("--n-estimators и --folds должны быть не меньше 2")

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=args.seed
    )
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    candidates = [1, 2, 4, 5, 8, 12, 16, 24, 30]
    rows = [
        evaluate_candidate(x_train, y_train, value, folds, args.n_estimators, args.seed)
        for value in candidates
    ]
    best = max(rows, key=lambda row: row["cv_roc_auc_mean"])

    # Тестовые данные используются только после выбора значения по кросс-валидации.
    final_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features=int(best["max_features"]),
        bootstrap=True,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train, y_train)
    test_prediction = final_model.predict(x_test)
    test_probability = final_model.predict_proba(x_test)[:, 1]
    test_accuracy = accuracy_score(y_test, test_prediction)
    test_auc = roc_auc_score(y_test, test_probability)
    test_confusion = confusion_matrix(y_test, test_prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, args.output_dir / "cv_results.csv")
    save_plot(rows, args.output_dir / "max_features.png")

    print(f"Данные: sklearn breast_cancer, {len(data.data)} объектов, {data.data.shape[1]} признаков")
    print(f"Разбиение: train={len(x_train)}, test={len(x_test)}; CV={args.folds} фолдов")
    print("\nmax_features  CV accuracy       CV ROC-AUC       corr деревьев")
    for row in rows:
        print(
            f"{row['max_features']:>12}  "
            f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}  "
            f"{row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}  "
            f"{row['tree_correlation_mean']:.4f}"
        )
    print(f"\nЛучшее max_features по CV ROC-AUC: {best['max_features']}")
    print(f"Отложенный тест: accuracy={test_accuracy:.4f}, ROC-AUC={test_auc:.4f}")
    print("Матрица ошибок (строки: истинный класс 0/1; столбцы: прогноз 0/1):")
    print(test_confusion)
    print(f"Результаты: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

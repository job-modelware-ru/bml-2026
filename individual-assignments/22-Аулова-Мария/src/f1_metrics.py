"""Macro-F1 и weighted-F1 на NumPy.

Строки матрицы ошибок соответствуют истинным классам, столбцы - предсказанным.
Порядок классов во всех результатах совпадает с порядком в `labels`.
"""
import numpy as np


def _check_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels должен быть непустым одномерным массивом")
    if np.unique(labels).size != labels.size:
        raise ValueError("labels содержит повторяющиеся значения")
    return labels


def _check_zero_division(zero_division):
    if zero_division not in (0, 1):
        raise ValueError("zero_division должен быть равен 0 или 1")
    return float(zero_division)


def _encode(y, labels, name):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} должен быть одномерным массивом")
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    pos = np.minimum(np.searchsorted(sorted_labels, y), labels.size - 1)
    known = sorted_labels[pos] == y
    if not known.all():
        raise ValueError(f"{name} содержит значения вне labels: {np.unique(y[~known])}")
    return order[pos]


def _divide(numerator, denominator, zero_division):
    out = np.full(numerator.shape, zero_division, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def confusion_matrix(y_true, y_pred, labels):
    """Матрица ошибок K x K для явно заданного набора меток."""
    labels = _check_labels(labels)
    true_idx = _encode(y_true, labels, "y_true")
    pred_idx = _encode(y_pred, labels, "y_pred")
    if true_idx.shape != pred_idx.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину")
    if true_idx.size == 0:
        raise ValueError("y_true и y_pred не должны быть пустыми")
    k = labels.size
    return np.bincount(true_idx * k + pred_idx, minlength=k * k).reshape(k, k)


def summarize_confusion_matrix(cm, zero_division=0):
    """Все метрики по готовой матрице ошибок (подходит и для синтетических матриц).

    Нулевой знаменатель у precision, recall и F1 заменяется на zero_division.
    F1 считается как 2TP / (2TP + FP + FN), поэтому он не определён
    только для класса с TP = FP = FN = 0.
    """
    zero_division = _check_zero_division(zero_division)
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError("cm должна быть непустой квадратной матрицей")
    if not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("cm должна содержать конечные неотрицательные значения")
    total = cm.sum()
    if total <= 0:
        raise ValueError("cm не должна быть нулевой")

    k = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = _divide(tp, tp + fp, zero_division)
    recall = _divide(tp, tp + fn, zero_division)
    f1 = _divide(2 * tp, 2 * tp + fp + fn, zero_division)

    shares = support / total
    macro = float(f1.mean())
    weighted = float(shares @ f1)
    contributions = (shares - 1.0 / k) * f1

    return {
        "confusion_matrix": cm,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(tp.sum() / total),
        "macro_f1": macro,
        "weighted_f1": weighted,
        "gap": weighted - macro,
        "gap_contributions": contributions,
    }


def f1_report(y_true, y_pred, labels, zero_division=0):
    """Матрица ошибок и все метрики по векторам меток."""
    labels = _check_labels(labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    report = summarize_confusion_matrix(cm, zero_division)
    report["labels"] = labels
    return report


def macro_f1(y_true, y_pred, labels, zero_division=0):
    return f1_report(y_true, y_pred, labels, zero_division)["macro_f1"]


def weighted_f1(y_true, y_pred, labels, zero_division=0):
    return f1_report(y_true, y_pred, labels, zero_division)["weighted_f1"]

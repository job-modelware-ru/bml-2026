"""Проверка f1_metrics.py. Запуск: python test_f1_metrics.py (или pytest)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from f1_metrics import (
    confusion_matrix,
    f1_report,
    macro_f1,
    summarize_confusion_matrix,
    weighted_f1,
)

ATOL = 1e-12
N_RANDOM_CASES = 2000
STATS = {"max_diff": 0.0, "sklearn_comparisons": 0, "absent_in_true": 0, "absent_in_pred": 0}


def _check_close(actual, desired):
    actual, desired = np.asarray(actual, dtype=float), np.asarray(desired, dtype=float)
    np.testing.assert_allclose(actual, desired, rtol=0, atol=ATOL)
    STATS["max_diff"] = max(STATS["max_diff"], float(np.max(np.abs(actual - desired))))


def assert_matches_sklearn(y_true, y_pred, labels, zero_division=0):
    report = f1_report(y_true, y_pred, labels, zero_division)
    kwargs = {"labels": labels, "zero_division": zero_division}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, **kwargs
    )
    np.testing.assert_array_equal(report["confusion_matrix"],
                                  sk_confusion_matrix(y_true, y_pred, labels=labels))
    np.testing.assert_array_equal(report["support"], support)
    _check_close(report["precision"], precision)
    _check_close(report["recall"], recall)
    _check_close(report["f1"], f1)
    _check_close(report["macro_f1"], f1_score(y_true, y_pred, average="macro", **kwargs))
    _check_close(report["weighted_f1"], f1_score(y_true, y_pred, average="weighted", **kwargs))
    _check_close(report["accuracy"], accuracy_score(y_true, y_pred))
    STATS["sklearn_comparisons"] += 1
    return report


def random_case(rng):
    k = int(rng.integers(2, 13))
    n = int(rng.integers(max(10, k), 600))
    alpha = float(rng.choice([0.2, 1.0, 5.0, 50.0]))
    labels = rng.permutation(rng.choice(200, size=k, replace=False))
    y_true = rng.choice(labels, size=n, p=rng.dirichlet(np.full(k, alpha)))
    noise = rng.choice(labels, size=n, p=rng.dirichlet(np.full(k, alpha)))
    y_pred = np.where(rng.random(n) < rng.uniform(0, 1), y_true, noise)
    return y_true, y_pred, labels


def expand(cm, labels):
    """Векторы меток, которые порождают заданную целочисленную матрицу ошибок."""
    rows, cols = np.indices(cm.shape)
    counts = cm.ravel()
    return np.repeat(labels[rows.ravel()], counts), np.repeat(labels[cols.ravel()], counts)


def test_random_cases_match_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(N_RANDOM_CASES):
        y_true, y_pred, labels = random_case(rng)
        assert_matches_sklearn(y_true, y_pred, labels)
        STATS["absent_in_true"] += int(np.setdiff1d(labels, y_true).size > 0)
        STATS["absent_in_pred"] += int(np.setdiff1d(labels, y_pred).size > 0)
    assert STATS["absent_in_true"] > 0 and STATS["absent_in_pred"] > 0


def test_imbalanced_majority_classifier():
    y_true = np.array([0] * 950 + [1] * 50)
    y_pred = np.zeros(1000, dtype=int)
    report = assert_matches_sklearn(y_true, y_pred, [0, 1])
    f1_major = 2 * 950 / (2 * 950 + 50)
    _check_close(report["f1"], [f1_major, 0.0])
    _check_close(report["macro_f1"], f1_major / 2)
    _check_close(report["weighted_f1"], 0.95 * f1_major)
    _check_close(report["accuracy"], 0.95)
    assert report["weighted_f1"] > report["macro_f1"]


def test_class_absent_from_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])
    report = assert_matches_sklearn(y_true, y_pred, [0, 1, 2])
    assert report["tp"][2] == 0 and report["fp"][2] == 0 and report["fn"][2] == 4
    _check_close([report["precision"][2], report["recall"][2], report["f1"][2]], [0, 0, 0])
    _check_close(report["macro_f1"], report["f1"].mean())


def test_class_absent_from_true():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 2, 1, 1, 2, 1, 0])
    report = assert_matches_sklearn(y_true, y_pred, [0, 1, 2])
    assert report["support"][2] == 0 and report["fp"][2] == 2
    _check_close([report["precision"][2], report["recall"][2], report["f1"][2]], [0, 0, 0])
    _check_close(report["gap_contributions"][2], 0.0)
    shares = report["support"][:2] / report["support"].sum()
    _check_close(report["weighted_f1"], shares @ report["f1"][:2])
    _check_close(report["macro_f1"], report["f1"][:2].sum() / 3)


def test_class_absent_from_both_but_in_labels():
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 2])
    labels = [0, 1, 2, 99]
    report = assert_matches_sklearn(y_true, y_pred, labels)
    assert report["support"][3] == 0 and report["tp"][3] + report["fp"][3] + report["fn"][3] == 0
    _check_close(report["f1"][3], 0.0)
    present = f1_report(y_true, y_pred, [0, 1, 2])
    _check_close(report["macro_f1"], present["macro_f1"] * 3 / 4)
    _check_close(report["weighted_f1"], present["weighted_f1"])
    report_one = assert_matches_sklearn(y_true, y_pred, labels, zero_division=1)
    _check_close(report_one["f1"][3], 1.0)
    _check_close(report_one["macro_f1"], (present["f1"].sum() + 1.0) / 4)
    _check_close(report_one["weighted_f1"], present["weighted_f1"])


def test_perfect_and_worst_predictions():
    labels = np.array([3, 1, 4, 7])
    rng = np.random.default_rng(1)
    y_true = rng.choice(labels, size=200, p=[0.6, 0.25, 0.1, 0.05])
    perfect = assert_matches_sklearn(y_true, y_true, labels)
    _check_close([perfect["macro_f1"], perfect["weighted_f1"], perfect["gap"]], [1, 1, 0])
    ordered = np.sort(labels)
    cyclic = ordered[(np.searchsorted(ordered, y_true) + 1) % labels.size]
    worst = assert_matches_sklearn(y_true, cyclic, labels)
    assert worst["tp"].sum() == 0
    _check_close([worst["macro_f1"], worst["weighted_f1"], worst["accuracy"]], [0, 0, 0])


def test_gap_identity():
    rng = np.random.default_rng(2)
    for _ in range(500):
        y_true, y_pred, labels = random_case(rng)
        report = f1_report(y_true, y_pred, labels)
        shares = report["support"] / report["support"].sum()
        covariance = np.mean((shares - shares.mean()) * (report["f1"] - report["f1"].mean()))
        k = labels.size
        _check_close(report["weighted_f1"] - report["macro_f1"], report["gap"])
        _check_close(report["gap"], report["gap_contributions"].sum())
        _check_close(report["gap"], k * covariance)


def test_equal_support_gives_equal_macro_and_weighted():
    rng = np.random.default_rng(3)
    for k, per_class in [(2, 40), (5, 25), (10, 30)]:
        labels = np.arange(k)
        y_true = np.repeat(labels, per_class)
        y_pred = np.where(rng.random(y_true.size) < 0.6, y_true, rng.integers(0, k, y_true.size))
        report = assert_matches_sklearn(y_true, y_pred, labels)
        _check_close(report["macro_f1"], report["weighted_f1"])
        _check_close(report["gap"], 0.0)
        _check_close(report["gap_contributions"].sum(), 0.0)


def test_weighted_below_macro():
    cm = np.array([[200, 200, 0], [200, 200, 0], [0, 0, 20]])
    report = summarize_confusion_matrix(cm)
    _check_close(report["f1"], [0.5, 0.5, 1.0])
    _check_close(report["macro_f1"], 2 / 3)
    _check_close(report["weighted_f1"], 420 / 820)
    assert report["weighted_f1"] < report["macro_f1"] and report["gap"] < 0
    labels = np.array([0, 1, 2])
    y_true, y_pred = expand(cm, labels)
    from_labels = assert_matches_sklearn(y_true, y_pred, labels)
    _check_close(from_labels["gap"], report["gap"])


def test_confusion_matrix_layout_and_label_order():
    y_true = np.array(["b", "b", "a", "c", "c", "c"])
    y_pred = np.array(["a", "b", "a", "c", "b", "c"])
    cm = confusion_matrix(y_true, y_pred, ["c", "a", "b"])
    np.testing.assert_array_equal(cm, [[2, 0, 1], [0, 1, 0], [0, 1, 1]])
    assert_matches_sklearn(y_true, y_pred, ["c", "a", "b"])
    forward = f1_report(y_true, y_pred, ["a", "b", "c"])
    backward = f1_report(y_true, y_pred, ["c", "b", "a"])
    _check_close(forward["f1"], backward["f1"][::-1])
    _check_close(forward["macro_f1"], backward["macro_f1"])
    _check_close(forward["weighted_f1"], backward["weighted_f1"])


def test_wrappers_agree_with_report():
    y_true, y_pred, labels = random_case(np.random.default_rng(4))
    report = f1_report(y_true, y_pred, labels)
    _check_close(macro_f1(y_true, y_pred, labels), report["macro_f1"])
    _check_close(weighted_f1(y_true, y_pred, labels), report["weighted_f1"])


def _raises(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_input_validation():
    y = np.array([0, 1, 1])
    assert _raises(f1_report, y, y[:2], [0, 1])
    assert _raises(f1_report, y, y, [0, 1, 1])
    assert _raises(f1_report, y, y, [])
    assert _raises(f1_report, y, y, [0])
    assert _raises(f1_report, y, np.array([0, 1, 2]), [0, 1])
    assert _raises(f1_report, y.reshape(3, 1), y, [0, 1])
    assert _raises(f1_report, y[:0], y[:0], [0, 1])
    assert _raises(f1_report, y, y, [0, 1], zero_division=0.5)
    assert _raises(summarize_confusion_matrix, np.ones((2, 3)))
    assert _raises(summarize_confusion_matrix, np.array([[1, -1], [0, 1]]))
    assert _raises(summarize_confusion_matrix, np.zeros((2, 2)))


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print()
    print(f"Пройдено тестов: {len(tests)} из {len(tests)}")
    print(f"Сравнений со sklearn: {STATS['sklearn_comparisons']} "
          f"(из них случайных: {N_RANDOM_CASES})")
    print(f"Случайные примеры с классом, отсутствующим в y_true: {STATS['absent_in_true']}, "
          f"в y_pred: {STATS['absent_in_pred']}")
    print(f"Максимальное абсолютное расхождение со sklearn: {STATS['max_diff']:.3e}")


if __name__ == "__main__":
    main()

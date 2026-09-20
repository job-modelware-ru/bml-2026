"""
Учебная реализация градиентного бустинга для регрессии.

Ограничения этой версии сделаны специально, чтобы алгоритм было легко
объяснить на презентации:
  * используются только числовые признаки;
  * базовые модели — решающие пни, то есть деревья глубины 1;
  * функция потерь — среднеквадратичная ошибка;
  * sklearn не используется.

Запуск:
    python gradient_boosting_from_scratch.py
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


def mean(values: Sequence[float]) -> float:
    """Среднее арифметическое."""
    if not values:
        raise ValueError("Нельзя вычислить среднее для пустого списка")
    return sum(values) / len(values)


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Среднеквадратичная ошибка."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true и y_pred должны иметь одинаковую длину")
    if not y_true:
        raise ValueError("Нельзя вычислить MSE для пустого набора")
    return sum((actual - predicted) ** 2 for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Корень из среднеквадратичной ошибки."""
    return mse(y_true, y_pred) ** 0.5


class RegressionStump:
    """Решающий пень для регрессии.

    Пень выбирает один признак и один порог:

        если X[j] <= threshold, вернуть left_value,
        иначе вернуть right_value.

    Значения в листьях равны средним целевых значений в соответствующих
    группах. Лучший порог выбирается по минимальной сумме квадратов ошибок.
    """

    def __init__(self) -> None:
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left_value: float = 0.0
        self.right_value: float = 0.0
        self.constant_value: float = 0.0

    def fit(self, X: Matrix, target: Sequence[float]) -> "RegressionStump":
        """Обучить пень на X и target."""
        if not X:
            raise ValueError("X не должен быть пустым")
        if len(X) != len(target):
            raise ValueError("X и target должны иметь одинаковую длину")
        if not X[0]:
            raise ValueError("В X должен быть хотя бы один признак")

        n_features = len(X[0])
        if any(len(row) != n_features for row in X):
            raise ValueError("Все строки X должны иметь одинаковое число признаков")

        self.constant_value = mean(target)
        best_error = sum((value - self.constant_value) ** 2 for value in target)
        best_split: Optional[Tuple[int, float, float, float]] = None

        for feature_index in range(n_features):
            unique_values = sorted({row[feature_index] for row in X})
            thresholds = [
                (left + right) / 2.0
                for left, right in zip(unique_values, unique_values[1:])
            ]

            for threshold in thresholds:
                left_indices = [
                    i for i, row in enumerate(X) if row[feature_index] <= threshold
                ]
                right_indices = [
                    i for i, row in enumerate(X) if row[feature_index] > threshold
                ]

                if not left_indices or not right_indices:
                    continue

                left_value = mean([target[i] for i in left_indices])
                right_value = mean([target[i] for i in right_indices])

                error = sum(
                    (target[i] - left_value) ** 2 for i in left_indices
                ) + sum(
                    (target[i] - right_value) ** 2 for i in right_indices
                )

                if error < best_error:
                    best_error = error
                    best_split = (
                        feature_index,
                        threshold,
                        left_value,
                        right_value,
                    )

        if best_split is None:
            # Если разделение не улучшает ошибку, пень возвращает константу.
            self.feature_index = None
            self.threshold = None
            self.left_value = self.constant_value
            self.right_value = self.constant_value
        else:
            (
                self.feature_index,
                self.threshold,
                self.left_value,
                self.right_value,
            ) = best_split

        return self

    def predict_one(self, row: Sequence[float]) -> float:
        """Получить предсказание для одного объекта."""
        if self.feature_index is None or self.threshold is None:
            return self.constant_value

        if row[self.feature_index] <= self.threshold:
            return self.left_value
        return self.right_value

    def predict(self, X: Matrix) -> Vector:
        """Получить предсказания для нескольких объектов."""
        return [self.predict_one(row) for row in X]

    def description(self) -> str:
        """Текстовое описание обученного пня для вывода на экран."""
        if self.feature_index is None or self.threshold is None:
            return f"константа {self.constant_value:.3f}"
        return (
            f"x[{self.feature_index}] <= {self.threshold:.3f}: "
            f"{self.left_value:.3f}, иначе {self.right_value:.3f}"
        )


class GradientBoostingRegressorScratch:
    """Градиентный бустинг над решающими пнями.

    Для MSE отрицательный градиент равен остатку:

        residual_i = y_i - F(x_i)

    На каждом шаге новый пень приближает residual, после чего его вклад
    добавляется к текущему предсказанию с коэффициентом learning_rate.
    """

    def __init__(self, n_estimators: int = 30, learning_rate: float = 0.1) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators должен быть положительным")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate должен быть в диапазоне (0, 1]")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_prediction_: Optional[float] = None
        self.estimators_: List[RegressionStump] = []
        self.train_mse_: List[float] = []

    def fit(self, X: Matrix, y: Sequence[float]) -> "GradientBoostingRegressorScratch":
        """Обучить модель."""
        if not X:
            raise ValueError("X не должен быть пустым")
        if len(X) != len(y):
            raise ValueError("X и y должны иметь одинаковую длину")

        self.initial_prediction_ = mean(y)
        current_predictions = [self.initial_prediction_] * len(y)
        self.estimators_ = []
        self.train_mse_ = []

        for _ in range(self.n_estimators):
            # Для MSE это отрицательный градиент функции потерь.
            residuals = [actual - predicted for actual, predicted in zip(y, current_predictions)]

            stump = RegressionStump().fit(X, residuals)
            residual_predictions = stump.predict(X)

            current_predictions = [
                predicted + self.learning_rate * correction
                for predicted, correction in zip(current_predictions, residual_predictions)
            ]

            self.estimators_.append(stump)
            self.train_mse_.append(mse(y, current_predictions))

        return self

    def predict(self, X: Matrix) -> Vector:
        """Получить предсказания обученной модели."""
        if self.initial_prediction_ is None:
            raise RuntimeError("Сначала вызовите fit()")

        predictions = [self.initial_prediction_] * len(X)
        for stump in self.estimators_:
            for i, stump_prediction in enumerate(stump.predict(X)):
                predictions[i] += self.learning_rate * stump_prediction
        return predictions


def train_test_split_manual(
    X: Matrix,
    y: Sequence[float],
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Matrix, Vector, Matrix, Vector]:
    """Простое разбиение данных без использования sklearn."""
    if len(X) != len(y):
        raise ValueError("X и y должны иметь одинаковую длину")
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1")

    indices = list(range(len(X)))
    random.Random(random_state).shuffle(indices)
    test_count = max(1, int(round(len(X) * test_size)))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = [X[i] for i in train_indices]
    y_train = [float(y[i]) for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [float(y[i]) for i in test_indices]
    return X_train, y_train, X_test, y_test


def main() -> None:
    # Небольшой искусственный набор данных для демонстрации регрессии.
    X = [[float(value)] for value in range(1, 13)]
    y = [1.0, 1.4, 1.8, 2.7, 3.0, 3.8, 4.2, 5.1, 5.4, 6.3, 6.7, 7.5]

    X_train, y_train, X_test, y_test = train_test_split_manual(X, y)

    model = GradientBoostingRegressorScratch(
        n_estimators=40,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("Градиентный бустинг, реализованный вручную")
    print("=" * 48)
    print(f"Начальное предсказание F0: {model.initial_prediction_:.3f}")
    print(f"Количество деревьев: {model.n_estimators}")
    print(f"Скорость обучения: {model.learning_rate}")
    print()

    print("Первые три решающих пня:")
    for number, stump in enumerate(model.estimators_[:3], start=1):
        print(f"  Дерево {number}: {stump.description()}")
    print()

    print(f"MSE на обучающей выборке: {mse(y_train, train_predictions):.4f}")
    print(f"RMSE на обучающей выборке: {rmse(y_train, train_predictions):.4f}")
    print(f"MSE на тестовой выборке:   {mse(y_test, test_predictions):.4f}")
    print(f"RMSE на тестовой выборке:  {rmse(y_test, test_predictions):.4f}")
    print()

    print("Тестовые предсказания:")
    for features, actual, predicted in zip(X_test, y_test, test_predictions):
        print(
            f"  x={features[0]:.1f} | "
            f"реальное y={actual:.1f} | "
            f"предсказание={predicted:.3f}"
        )

    print()
    print("MSE после первых итераций обучения:")
    for iteration in (1, 2, 3, 5, 10, 20, 40):
        print(f"  Итерация {iteration:2d}: {model.train_mse_[iteration - 1]:.4f}")


if __name__ == "__main__":
    main()

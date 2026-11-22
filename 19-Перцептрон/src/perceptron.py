"""Перцептрон Розенблатта для бинарной классификации.

Модель реализована самостоятельно, без готовых ML-библиотек.
Из внешних зависимостей используется только numpy.
"""

import numpy as np


class Perceptron:
    """Простейший перцептрон со ступенчатой функцией активации.

    Классы должны быть закодированы числами 0 и 1.

    Для объекта x вычисляется взвешенная сумма

        z = w1*x1 + w2*x2 + ... + wn*xn + b,

    после чего применяется ступенчатая функция:

        y_hat = 1, если z >= 0,
        y_hat = 0, если z < 0.

    Если предсказание не совпало с правильным ответом, веса обновляются:

        w_i = w_i + eta * (y - y_hat) * x_i,
        b   = b   + eta * (y - y_hat).
    """

    def __init__(self, learning_rate=0.1, n_epochs=100):
        """Создает перцептрон.

        Параметры:
            learning_rate : скорость обучения eta (по умолчанию 0.1);
            n_epochs      : максимальное число эпох обучения.
        """
        if learning_rate <= 0:
            raise ValueError("Скорость обучения должна быть больше нуля.")
        if n_epochs <= 0:
            raise ValueError("Количество эпох должно быть больше нуля.")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Эти поля заполняются в методе fit.
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []
        self.n_epochs_run = 0
        self.converged = False

    def net_input(self, X):
        """Возвращает взвешенную сумму z = X @ w + b."""
        if self.weights is None:
            raise RuntimeError("Модель еще не обучена. Сначала вызовите fit().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X должен быть вектором или двумерной матрицей.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Ожидалось {self.weights.shape[0]} признаков, "
                f"а получено {X.shape[1]}."
            )

        return X @ self.weights + self.bias

    def predict(self, X):
        """Возвращает предсказанные классы (0 или 1)."""
        # Ступенчатая активация: 1 при z >= 0, иначе 0.
        return (self.net_input(X) >= 0).astype(int)

    def fit(self, X, y, verbose=False):
        """Обучает модель по правилу обновления весов Розенблатта.

        Параметры:
            X       : матрица признаков размера (n_objects, n_features);
            y       : вектор правильных ответов из нулей и единиц;
            verbose : если True, печатает вычисления для каждого объекта.

        Возвращает сам объект Perceptron.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(
                "X должен быть двумерным массивом: "
                "строки — объекты, столбцы — признаки."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError("Количество объектов в X и y должно совпадать.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Метки классов должны быть 0 или 1.")

        y = y.astype(int)

        # Начальные веса — нули. Для линейно разделимых данных
        # перцептрон сходится к разделяющей гиперплоскости.
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_per_epoch = []
        self.n_epochs_run = 0
        self.converged = False

        for epoch in range(1, self.n_epochs + 1):
            errors = 0

            for i in range(X.shape[0]):
                x = X[i]
                target = y[i]

                z = np.dot(x, self.weights) + self.bias
                y_hat = 1 if z >= 0 else 0
                error = target - y_hat

                if verbose:
                    x_str = "(" + ", ".join(f"{v:g}" for v in x) + ")"
                    print(
                        f"  объект {i + 1}: x={x_str}, z={z:.4f}, "
                        f"y_hat={y_hat}, error={error}",
                        end="",
                    )

                # Веса меняются только при ошибке.
                if error != 0:
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
                    errors += 1

                    if verbose:
                        w_str = "(" + ", ".join(
                            f"{v:.4f}" for v in self.weights
                        ) + ")"
                        print(f" -> w={w_str}, b={self.bias:.4f}")
                elif verbose:
                    print(" -> веса без изменений")

            self.errors_per_epoch.append(errors)
            self.n_epochs_run = epoch

            if verbose:
                print(f"  эпоха {epoch}: ошибок {errors}")

            # Если за всю эпоху не было ни одной ошибки,
            # дальнейшее обучение ничего не изменит.
            if errors == 0:
                self.converged = True
                break

        return self

    def score(self, X, y):
        """Возвращает долю правильных ответов."""
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

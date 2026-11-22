"""Пошаговый численный пример работы перцептрона.

Скрипт показывает:
1) один шаг обучения на конкретных числах;
2) тот же объект, но с правильным классом 0 (чтобы увидеть обновление весов);
3) полный процесс обучения на наборе AND.

Запуск:
    python numerical_example.py
"""

import numpy as np

from perceptron import Perceptron


def print_header(title):
    """Печатает заголовок раздела."""
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def activation(z):
    """Ступенчатая функция активации."""
    return 1 if z >= 0 else 0


def one_step(x1, x2, w1, w2, b, y, eta):
    """Выполняет один шаг перцептрона и печатает все вычисления."""
    print(f"Входные данные: x1 = {x1:g}, x2 = {x2:g}")
    print(f"Начальные параметры: w1 = {w1:g}, w2 = {w2:g}, b = {b:g}")
    print(f"Целевой класс: y = {y}, скорость обучения: eta = {eta:g}")
    print()

    z = w1 * x1 + w2 * x2 + b

    print("Шаг 1. Взвешенная сумма:")
    print("  z = w1*x1 + w2*x2 + b")
    print(f"  z = {w1:g}*{x1:g} + {w2:g}*{x2:g} + {b:g}")
    print(f"  z = {w1 * x1:g} + {w2 * x2:g} + {b:g} = {z:g}")
    print()

    y_hat = activation(z)

    print("Шаг 2. Функция активации:")
    sign = ">=" if z >= 0 else "<"
    print(f"  z = {z:g} {sign} 0, поэтому y_hat = {y_hat}")
    print()

    error = y - y_hat

    print("Шаг 3. Ошибка:")
    print(f"  error = y - y_hat = {y} - {y_hat} = {error}")
    print()

    print("Шаг 4. Обновление весов:")
    print("  w_i_new = w_i + eta * error * x_i")
    print("  b_new   = b + eta * error")

    if error == 0:
        w1_new, w2_new, b_new = w1, w2, b
        print("  error = 0, поэтому веса и смещение не меняются.")
    else:
        dw1 = eta * error * x1
        dw2 = eta * error * x2
        db = eta * error

        w1_new = w1 + dw1
        w2_new = w2 + dw2
        b_new = b + db

        print(f"  dw1 = {eta:g} * ({error}) * {x1:g} = {dw1:g}")
        print(f"  dw2 = {eta:g} * ({error}) * {x2:g} = {dw2:g}")
        print(f"  db  = {eta:g} * ({error}) = {db:g}")
        print(f"  w1_new = {w1:g} + ({dw1:g}) = {w1_new:g}")
        print(f"  w2_new = {w2:g} + ({dw2:g}) = {w2_new:g}")
        print(f"  b_new  = {b:g} + ({db:g}) = {b_new:g}")

    print()
    return w1_new, w2_new, b_new


def manual_example():
    """Ручной расчет для одного объекта."""
    print_header("ЧИСЛЕННЫЙ ПРИМЕР: ОДИН ШАГ ОБУЧЕНИЯ")

    print("Рассмотрим перцептрон с двумя входами.")
    print("Пусть объект уже правильно классифицирован.")
    print()

    one_step(0.5, 0.3, 0.2, 0.4, 0.1, 1, 0.1)

    print("Итог: error = 0, значит, веса остаются прежними.")
    print("Так работает перцептрон, когда ответ уже правильный.")
    print()

    print_header("ТОТ ЖЕ ОБЪЕКТ, НО ПРАВИЛЬНЫЙ КЛАСС 0")

    print("Теперь представим, что тот же вход относится к классу 0.")
    print("Перцептрон ошибется, и веса изменятся.")
    print()

    one_step(0.5, 0.3, 0.2, 0.4, 0.1, 0, 0.1)

    print("Итог: error = -1, поэтому веса уменьшились.")
    print("Один шаг не обязан сразу исправить объект —")
    print("обучение продолжается на следующих объектах и эпохах.")


def and_example():
    """Обучение перцептрона на наборе AND с подробным выводом."""
    print_header("ОБУЧЕНИЕ НА НАБОРЕ AND")

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1])

    print("X — входные признаки, y — правильные ответы:")
    print("X =")
    print(X)
    print("y =", y)
    print()
    print("Начальные веса: w = (0, 0), b = 0")
    print("Скорость обучения: eta = 0.5")
    print("Порядок объектов: (0,0), (0,1), (1,0), (1,1)")
    print("Берем eta = 0.5, чтобы веса и суммы в таблице были круглыми.")
    print()

    model = Perceptron(learning_rate=0.5, n_epochs=20)
    model.fit(X, y, verbose=True)

    print()
    print(f"Обучение заняло эпох: {model.n_epochs_run}")
    print(f"Итоговые веса: w = {np.round(model.weights, 4)}, "
          f"b = {model.bias:.4f}")

    if model.converged:
        print("Модель сошлась: за целую эпоху не было ни одной ошибки.")
    else:
        print("За отведенное число эпох модель не сошлась.")

    print()
    print("Проверка на обучающей выборке:")
    for x, target in zip(X, y):
        prediction = int(model.predict(x)[0])
        result = "верно" if prediction == target else "ошибка"
        print(f"  x = ({x[0]:g}, {x[1]:g}): "
              f"y_hat = {prediction}, y = {target} — {result}")

    print(f"Точность: {model.score(X, y):.2f}")


def main():
    manual_example()
    and_example()
    print()


if __name__ == "__main__":
    main()

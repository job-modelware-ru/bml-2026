from knn import classify, regress


CLASS_NAMES = ["A1", "A2", "A3", "B1", "B2", "B3"]
CLASS_X = [
    [1, 2],
    [2, 3],
    [2, 1],
    [6, 4],
    [7, 5],
    [6, 6],
]
CLASS_Y = ["A", "A", "A", "B", "B", "B"]

FLAT_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
FLAT_X = [[50], [60], [68], [74], [76], [85], [95]]
FLAT_Y = [4.8, 5.7, 6.6, 7.2, 7.5, 8.4, 9.3]


def read_int(text, default):
    value = input(f"{text} [{default}]: ").strip()
    return default if not value else int(value)


def read_numbers(text, default):
    default_text = " ".join(str(x) for x in default)
    value = input(f"{text} [{default_text}]: ").strip()

    if not value:
        return default

    return [float(x.replace(",", ".")) for x in value.split()]


def read_weights():
    value = input("Веса: uniform / distance [uniform]: ").strip().lower()
    return value or "uniform"


def classification():
    k = read_int("k", 3)
    point = read_numbers("Координаты X", [3, 2])
    weights = read_weights()

    if len(point) != 2:
        raise ValueError("Нужно ввести две координаты")

    prediction, neighbors, votes = classify(
        CLASS_X,
        CLASS_Y,
        point,
        k,
        weights,
        CLASS_NAMES,
    )

    print(f"\nX = {tuple(point)}, k = {k}")
    print("Ближайшие соседи:")

    for i, neighbor in enumerate(neighbors, 1):
        print(
            f"{i}. {neighbor['name']} "
            f"{tuple(neighbor['point'])} "
            f"класс {neighbor['target']}, "
            f"d = {neighbor['distance']:.3f}"
        )

    if weights == "uniform":
        result = ", ".join(f"{key}={int(value)}" for key, value in votes.items())
        print(f"Голоса: {result}")
    else:
        result = ", ".join(f"{key}={value:.3f}" for key, value in votes.items())
        print(f"Веса: {result}")

    print(f"Прогноз: класс {prediction}")


def regression():
    k = read_int("k", 3)
    point = read_numbers("Площадь квартиры", [72])
    weights = read_weights()

    if len(point) != 1:
        raise ValueError("Нужно ввести одну площадь")

    prediction, neighbors = regress(
        FLAT_X,
        FLAT_Y,
        point,
        k,
        weights,
        FLAT_NAMES,
    )

    print(f"\nПлощадь = {point[0]:g} м², k = {k}")
    print("Ближайшие квартиры:")

    for i, neighbor in enumerate(neighbors, 1):
        print(
            f"{i}. {neighbor['point'][0]:g} м², "
            f"{neighbor['target']:.1f} млн ₽, "
            f"d = {neighbor['distance']:.3f}"
        )

    print(f"Прогноз: {prediction:.3f} млн ₽")


def main():
    while True:
        print("\n1 — классификация")
        print("2 — регрессия")
        print("0 — выход")

        choice = input("Выбор [1]: ").strip() or "1"

        if choice == "0":
            break

        try:
            if choice == "1":
                classification()
            elif choice == "2":
                regression()
            else:
                print("Нет такого пункта")
        except ValueError as error:
            print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()

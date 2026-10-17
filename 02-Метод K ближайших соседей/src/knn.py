from math import sqrt


def euclidean_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Размерности объектов должны совпадать")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def get_neighbors(x_train, y_train, point, k, names=None):
    if len(x_train) != len(y_train):
        raise ValueError("Количество объектов и ответов не совпадает")
    if not 1 <= k <= len(x_train):
        raise ValueError("k должно быть от 1 до количества объектов")

    neighbors = []

    for i, (x, y) in enumerate(zip(x_train, y_train)):
        neighbors.append({
            "name": names[i] if names else str(i),
            "point": x,
            "target": y,
            "distance": euclidean_distance(x, point),
        })

    neighbors.sort(key=lambda item: item["distance"])
    return neighbors[:k]


def classify(x_train, y_train, point, k=3, weights="uniform", names=None):
    neighbors = get_neighbors(x_train, y_train, point, k, names)

    if weights not in ("uniform", "distance"):
        raise ValueError('weights: "uniform" или "distance"')

    votes = {}

    if weights == "distance":
        exact = [n for n in neighbors if n["distance"] == 0]

        if exact:
            for neighbor in exact:
                label = neighbor["target"]
                votes[label] = votes.get(label, 0) + 1
        else:
            for neighbor in neighbors:
                label = neighbor["target"]
                weight = 1 / neighbor["distance"]
                votes[label] = votes.get(label, 0) + weight
    else:
        for neighbor in neighbors:
            label = neighbor["target"]
            votes[label] = votes.get(label, 0) + 1

    prediction = max(votes, key=votes.get)
    return prediction, neighbors, votes


def regress(x_train, y_train, point, k=3, weights="uniform", names=None):
    neighbors = get_neighbors(x_train, y_train, point, k, names)

    if weights not in ("uniform", "distance"):
        raise ValueError('weights: "uniform" или "distance"')

    if weights == "uniform":
        prediction = sum(n["target"] for n in neighbors) / k
        return prediction, neighbors

    exact = [n for n in neighbors if n["distance"] == 0]
    if exact:
        prediction = sum(n["target"] for n in exact) / len(exact)
        return prediction, neighbors

    weighted_sum = 0
    total_weight = 0

    for neighbor in neighbors:
        weight = 1 / neighbor["distance"]
        weighted_sum += neighbor["target"] * weight
        total_weight += weight

    return weighted_sum / total_weight, neighbors

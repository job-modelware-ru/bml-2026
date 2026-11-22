import numpy as np


UNVISITED = -2
NOISE = -1


def dist_to_all(points: np.ndarray, index: int) -> np.ndarray:
    #возвращает евклидовы расстояния от точки index до всех точек
    diff = points - points[index]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def find_neighbours(points: np.ndarray, index: int, eps: float) -> list[int]:
    #возвращает индексы соседей точки index, включая саму точку index
    neighbours = []
    dists = dist_to_all(points, index)
    for i, dist in enumerate(dists):
        if dist <= eps:
            neighbours.append(i)
    return neighbours


def dbscan(points: np.ndarray, eps: float, min_samples: int) -> tuple[list[int], int]:
    #реализация DBSCAN
    labels = [UNVISITED] * len(points)
    cluster_count = 0

    for i in range(len(points)):
        if labels[i] != UNVISITED:
            continue

        neighbours = find_neighbours(points, i, eps)
        if len(neighbours) < min_samples:
            labels[i] = NOISE
            continue

        labels[i] = cluster_count
        stack = []

        for neighbour in neighbours:
            if labels[neighbour] == UNVISITED:
                labels[neighbour] = cluster_count
                stack.append(neighbour)
            elif labels[neighbour] == NOISE:
                labels[neighbour] = cluster_count

        while stack:
            current_index = stack.pop()
            next_neighbours = find_neighbours(points, current_index, eps)
            if len(next_neighbours) < min_samples:
                continue

            for neighbour in next_neighbours:
                if labels[neighbour] == UNVISITED:
                    labels[neighbour] = cluster_count
                    stack.append(neighbour)
                elif labels[neighbour] == NOISE:
                    labels[neighbour] = cluster_count

        cluster_count += 1

    return labels, cluster_count

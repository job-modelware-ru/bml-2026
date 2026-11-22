import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dbscan import NOISE, dbscan, find_neighbours


def choose_dataset(project_dir: Path) -> Path:
    #показывает csv-файлы в папке программы и возвращает путь к выбранному файлу
    files = sorted(project_dir.glob("*.csv"))
    print("доступные наборы данных:")
    for number, path in enumerate(files, start=1):
        print(f"{number}. {path.name}")
    number = int(input("выберите номер набора: "))
    return files[number - 1]


def load_points(path: Path) -> np.ndarray:
    #читает csv-файл с заголовком x,y и возвращает массив координат точек
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row])
    return np.array(rows, dtype=float)


def plot_result(points: np.ndarray, labels: list[int], eps: float, min_samples: int, output: Path, show: bool = True) -> None:
    #сохраняет график кластеров, выделяет типы точек и открывает окно графика
    fig, ax = plt.subplots(figsize=(11, 6))
    point_size = 65 if len(points) <= 30 else 14
    label_array = np.asarray(labels)
    is_core = np.array([len(find_neighbours(points, i, eps)) >= min_samples for i in range(len(points))], dtype=bool)
    for cluster in sorted(set(labels) - {NOISE}):
        color = plt.get_cmap("tab10")(cluster % 10)
        core_points = points[(label_array == cluster) & is_core]
        border_points = points[(label_array == cluster) & ~is_core]
        if len(core_points):
            ax.scatter(
                core_points[:, 0], core_points[:, 1], s=point_size,
                color=color, label=f"Кластер {cluster}: центральные",
            )
        if len(border_points):
            ax.scatter(
                border_points[:, 0], border_points[:, 1], s=point_size + 15,
                facecolors="none", edgecolors=color, linewidths=1.3,
                label=f"Кластер {cluster}: граничные",
            )
    noise = points[label_array == NOISE]
    if len(noise):
        ax.scatter(noise[:, 0], noise[:, 1], c="black", marker="x", s=85, label="Шум")
    ax.set_title(f"{len(points)} точек, eps = {eps}, min_samples = {min_samples}")
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    if len(points):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print("график сохранён:", output)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    #читает данные и параметры, запускает dbscan и строит график результата
    project_dir = Path(__file__).resolve().parent
    data_file = choose_dataset(project_dir)
    points = load_points(data_file)
    eps = float(input("введите eps: "))
    min_samples = int(input("введите min_samples: "))
    labels, _ = dbscan(points, eps, min_samples)
    print("использован набор данных:", data_file.name)
    output = project_dir / "results" / f"{data_file.stem}_eps_{eps}_min_{min_samples}.png"
    plot_result(points, labels, eps, min_samples, output)


if __name__ == "__main__":
    main()

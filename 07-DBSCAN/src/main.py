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
    #читает csv-файл без заголовка и возвращает массив координат точек
    return np.loadtxt(path, delimiter=",", ndmin=2)


def plot_result(points: np.ndarray, labels: list[int], eps: float, min_samples: int, output: Path, show: bool = True) -> None:
    #сохраняет график кластеров, выделяет типы точек и открывает окно графика
    figsize = (8, 6) if len(points) <= 30 else (11, 6)
    one_feature = points.shape[1] == 1
    if one_feature:
        figsize = (figsize[0], 2.4)
    fig, ax = plt.subplots(figsize=figsize)
    point_size = 65 if len(points) <= 30 else 14
    label_array = np.asarray(labels)
    is_core = np.array([len(find_neighbours(points, i, eps)) >= min_samples for i in range(len(points))], dtype=bool)
    plot_x = points[:, 0]
    plot_y = np.zeros(len(points)) if one_feature else points[:, 1]
    if one_feature:
        point_size = 12
    for cluster in sorted(set(labels) - {NOISE}):
        color = plt.get_cmap("tab10")(cluster % 10)
        core_mask = (label_array == cluster) & is_core
        border_mask = (label_array == cluster) & ~is_core
        if np.any(core_mask):
            ax.scatter(plot_x[core_mask], plot_y[core_mask], s=point_size, color=color, label=f"Кластер {cluster}: центральные")
        if np.any(border_mask):
            ax.scatter(plot_x[border_mask], plot_y[border_mask], s=point_size + 15, facecolors="none", edgecolors=color, linewidths=1.3, label=f"Кластер {cluster}: граничные")
    noise_mask = label_array == NOISE
    if np.any(noise_mask):
        ax.scatter(plot_x[noise_mask], plot_y[noise_mask], c="black", marker="x", s=30 if one_feature else 85, label="Шум")

    ax.set_xlabel("Признак 1")
    if one_feature:
        ax.yaxis.set_visible(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["bottom"].set_zorder(0.5)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.set_ylabel("Признак 2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    ax.set_title(f"{len(points)} точек, eps = {eps}, min_samples = {min_samples}")
    if points.shape[1] > 2:
        ax.set_title(f"Первые два признака из {points.shape[1]}, {len(points)} точек, eps = {eps}, min_samples = {min_samples}")
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
    point_names = ["A", "B", "C", "D", "E", "F", "G", "H"] if data_file.name == "points.csv" else None
    for cluster in sorted(set(labels) - {NOISE}):
        members = [point_names[i] if point_names else str(i + 1) for i, label in enumerate(labels) if label == cluster]
        print(f"кластер {cluster}: {', '.join(members)}")
    noise = [point_names[i] if point_names else str(i + 1) for i, label in enumerate(labels) if label == NOISE]
    if noise:
        print(f"шумовые точки: {', '.join(noise)}")
    output = project_dir / "results" / f"{data_file.stem}_eps_{eps}_min_{min_samples}.png"
    plot_result(points, labels, eps, min_samples, output)


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons


RANDOM_SEED = 42


def generate_datasets() -> dict[str, np.ndarray]:
    #генерирует три набора точек и возвращает словарь имён файлов и массивов координат
    moons, _ = make_moons(n_samples=700, noise=0.05, shuffle=True, random_state=RANDOM_SEED)
    moons *= 25

    rings, _ = make_circles(n_samples=800, noise=0.05, factor=0.5, shuffle=True, random_state=RANDOM_SEED)
    rings *= 20

    blobs, _ = make_blobs(
        n_samples=900,
        centers=[[-15, -10], [0, 12.5], [15, -5]],
        cluster_std=1.5, n_features=2,
        shuffle=True, random_state=RANDOM_SEED,
    )

    return {
        "points_moons.csv": moons,
        "points_rings.csv": rings,
        "points_blobs.csv": blobs,
    }


def main() -> None:
    #сохраняет сгенерированные наборы точек в csv-файлы
    project_dir = Path(__file__).resolve().parent
    for filename, points in generate_datasets().items():
        output = project_dir / filename
        np.savetxt(output, points, delimiter=",", fmt="%.6f")
        print(f"сохранён {filename}: {len(points)} точек")


if __name__ == "__main__":
    main()

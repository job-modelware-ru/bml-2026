"""Общие настройки: пути, оформление рисунков, воспроизводимость, вывод.

Модуль импортируется всеми скриптами каталога. Здесь собрано то, что
не относится к самой теме CNN, но нужно, чтобы результаты были
одинаковыми при каждом запуске и рисунки выглядели единообразно.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # рисуем в файлы, окно не открываем
import matplotlib.pyplot as plt

# Консоль Windows по умолчанию может не печатать символы вроде «∂» и «→»
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"          # сюда torchvision скачает Fashion-MNIST
FIG_DIR = HERE / "figures"        # рисунки для слайдов
RESULTS_DIR = HERE / "results"    # числа (json) и веса обученных моделей
for _d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Палитра (совпадает с палитрой презентации)
# ---------------------------------------------------------------------------
CNN_COLOR = "#e4572e"   # свёрточная сеть — всегда этот цвет
MLP_COLOR = "#2e86ab"   # полносвязная сеть
TEAL = "#1b998b"
AMBER = "#f2a541"
VIOLET = "#6c5b7b"
GREY = "#8a90a0"
SERIES = [CNN_COLOR, MLP_COLOR, TEAL, AMBER, VIOLET, GREY]

INK = "#1d2433"
INK_2 = "#4a5263"
MUTED = "#8a90a0"
GRID = "#e4e6eb"
SURFACE = "#ffffff"

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 11,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": "#c9ccd3",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "figure.dpi": 110,
}
plt.rcParams.update(RC_PARAMS)

# Названия классов Fashion-MNIST в порядке меток 0…9
CLASSES = ["Футболка", "Брюки", "Свитер", "Платье", "Пальто",
           "Сандалия", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]


# ---------------------------------------------------------------------------
# Воспроизводимость и устройство
# ---------------------------------------------------------------------------
def set_seed(seed=RANDOM_STATE):
    """Зафиксировать все генераторы случайных чисел.

    Случайность в обучении сети: начальные веса, порядок примеров в
    мини-батчах, маски Dropout. Без фиксации каждый запуск даёт немного
    другую точность, и числа на слайдах нельзя было бы воспроизвести.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    """Видеокарта, если она есть, иначе процессор."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------
def save(fig, name, svg=True):
    """Сохранить рисунок в figures/<name>.png (и .svg) и закрыть его.

    PNG вставляется в презентацию; SVG — векторная копия, которую можно
    править в Inkscape / Illustrator / PowerPoint без потери качества.
    """
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.15)
    if svg:
        fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  рисунок: figures/{png.name}")
    return png


def save_json(obj, name):
    """Сохранить словарь с числами в results/<name>.json."""
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  числа:   results/{path.name}")
    return path


def load_json(name):
    """Прочитать results/<name>.json (или None, если файла ещё нет)."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Оформление консольного вывода
# ---------------------------------------------------------------------------
def section(title):
    """Заголовок раздела в консоли."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def subsection(title):
    """Заголовок подраздела в консоли."""
    print(f"\n--- {title} " + "-" * max(0, 68 - len(title)))


def matrix_str(m, fmt="{:6.2f}", indent="    "):
    """Матрица в виде выровненных строк — для печати численного примера."""
    m = np.atleast_2d(m)
    return "\n".join(indent + " ".join(fmt.format(v) for v in row) for row in m)

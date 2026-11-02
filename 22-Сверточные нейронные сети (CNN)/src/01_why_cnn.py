"""Постановка задачи: изображение как данные и почему полносвязной сети трудно.

1. Набор Fashion-MNIST: 10 классов, примеры картинок.
2. Картинка — матрица яркостей 28×28 (числа от 0 до 255).
3. Для полносвязной сети картинка — вектор из 784 чисел. Сдвиг на пару
   пикселей не меняет смысла картинки, но сильно меняет этот вектор:
   измеряем косинусное сходство векторов до и после сдвига на всём тесте.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES, CNN_COLOR, INK, save, save_json, section
from data import raw_images

IMG_INDEX = 0          # тестовая картинка №0 — ботинок
N_PER_CLASS = 8


def plot_samples(images, labels):
    """Сетка: по строке на класс, N_PER_CLASS примеров в строке."""
    section("1. Набор данных Fashion-MNIST")
    counts = np.bincount(labels)
    print(f"  тестовых картинок: {len(labels)}, по классам: {counts.tolist()}")
    fig, axes = plt.subplots(10, N_PER_CLASS, figsize=(N_PER_CLASS * 0.9 + 1.4, 10 * 0.9))
    rng = np.random.RandomState(0)
    for c in range(10):
        idx = rng.choice(np.where(labels == c)[0], N_PER_CLASS, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(images[i], cmap="gray_r")
            ax.axis("off")
        axes[c, 0].text(-4, 14, f"{c}  {CLASSES[c]}", ha="right", va="center",
                        fontsize=11, color=INK)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.2)
    save(fig, "fashion_samples")


def plot_pixel_matrix(img):
    """Картинка целиком и увеличенный фрагмент 8×8 с числами."""
    section("2. Картинка как матрица чисел")
    r0, c0 = 15, 16  # фрагмент у каблука ботинка
    crop = img[r0:r0 + 8, c0:c0 + 8]
    print(f"  размер {img.shape}, тип {img.dtype}, яркости от {img.min()} до {img.max()}")
    print(f"  фрагмент [{r0}:{r0 + 8}, {c0}:{c0 + 8}]:\n{crop}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    a1.imshow(img, cmap="gray_r")
    a1.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), 8, 8, fill=False,
                               edgecolor=CNN_COLOR, lw=2.5))
    a1.set_title("28 × 28 = 784 пикселя", fontsize=12)
    a1.axis("off")
    a2.imshow(crop, cmap="gray_r", vmin=0, vmax=255)
    for i in range(8):
        for j in range(8):
            v = crop[i, j]
            a2.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="white" if v > 120 else INK)
    a2.set_title("фрагмент 8 × 8: яркость 0…255", fontsize=12)
    a2.set_xticks([]), a2.set_yticks([])
    for spine in a2.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(CNN_COLOR)
        spine.set_linewidth(2.5)
    a2.grid(False)
    fig.tight_layout()
    save(fig, "pixel_matrix")


def shift_images(x, dx):
    """Сдвинуть пачку картинок N×28×28 вправо на dx пикселей (слева — нули)."""
    out = np.zeros_like(x)
    out[:, :, dx:] = x[:, :, :28 - dx]
    return out


def cosine(a, b):
    """Косинус угла между векторами (по строкам)."""
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def pair_similarity(images, labels, n_pairs=20_000, seed=0):
    """Среднее косинусное сходство случайных пар: одного класса и разных."""
    flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed)
    a, b = rng.randint(0, len(flat), n_pairs), rng.randint(0, len(flat), n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    cos = cosine(flat[a], flat[b])
    same = labels[a] == labels[b]
    return float(cos[same].mean()), float(cos[~same].mean())


def shift_effect(images):
    """Насколько меняется вектор из 784 чисел при сдвиге картинки."""
    section("3. Сдвиг картинки глазами полносвязной сети")
    x = images.astype(np.float32) / 255.0
    flat = x.reshape(len(x), -1)
    rows = []
    for dx in range(0, 5):
        sh = shift_images(x, dx).reshape(len(x), -1)
        cos = cosine(flat, sh).mean()
        changed = (np.abs(sh - flat) > 0.25).mean(1).mean()
        rows.append({"shift": dx, "cosine": float(cos), "changed_share": float(changed)})
        print(f"  сдвиг {dx} пикс.: косинусное сходство {cos:.3f}, "
              f"меняется > 25 % яркости у {changed:.1%} пикселей")
    return rows


def plot_shift_vectors(img, rows, same_cls):
    """Картинка, её сдвиг на 2 пикселя и попиксельная разность.

    Для человека первые две картинки одинаковы. Полносвязная сеть сравнивает
    их попиксельно (вход i ↔ вход i), и для неё отличаются все пиксели
    третьей картинки.
    """
    x = img.astype(np.float32) / 255.0
    shifted = shift_images(x[None], 2)[0]
    diff = np.abs(shifted - x)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    panels = [(x, "исходная", "gray_r"), (shifted, "сдвиг на 2 пикселя вправо", "gray_r"),
              (diff, "|разность| попиксельно", "Reds")]
    for ax, (m, title, cmap) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor="#c9ccd3", lw=1))
    fig.tight_layout()
    save(fig, "shift_vectors")


def main():
    images, labels = raw_images(train=False)
    plot_samples(images, labels)
    plot_pixel_matrix(images[IMG_INDEX])
    rows = shift_effect(images)
    same_cls, diff_cls = pair_similarity(images, labels)
    print("\n  Для сравнения — случайные пары разных картинок:")
    print(f"    одного класса: {same_cls:.3f},  разных классов: {diff_cls:.3f}")
    worse = [r["shift"] for r in rows if r["cosine"] < same_cls]
    print(f"  Уже при сдвиге на {worse[0]} пикс. вещь становится для MLP «дальше» от себя,\n"
          f"  чем в среднем другая вещь того же класса.")
    plot_shift_vectors(images[IMG_INDEX], rows, same_cls)
    save_json({"shift_effect": rows, "cos_same_class": same_cls,
               "cos_diff_class": diff_cls}, "why_cnn")


if __name__ == "__main__":
    main()

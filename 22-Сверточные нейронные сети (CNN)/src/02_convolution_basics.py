"""Операции свёрточной сети «с нуля» на NumPy и их свойства.

1. Свёртка с отступом (padding) и шагом (stride): реализация, формула
   размера выхода, сверка с torch.nn.functional.conv2d.
2. Классические ядра (размытие, Собель, Лаплас) на картинке из Fashion-MNIST:
   одно и то же ядро 3×3 находит один и тот же признак в любом месте.
3. Число параметров свёрточного слоя в сравнении с полносвязным.
4. Эквивариантность к сдвигу: свёртка сдвинутой картинки = сдвинутая свёртка.
5. Рецептивное поле нейронов SimpleCNN.
Кроме того, строятся схемы для слайдов: операция свёртки, padding и stride,
пулинг (max и average), многоканальная свёртка.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from common import (CNN_COLOR, INK, INK_2, MLP_COLOR, MUTED, TEAL, matrix_str,
                    save, save_json, section, subsection)
from data import raw_images
from diagrams import arrow_between, draw_grid
from models import fmt_int

DIVERGING = plt.get_cmap("RdBu_r")
IMG_INDEX = 0  # тестовая картинка №0 — ботинок


# ---------------------------------------------------------------------------
# 1. Свёртка
# ---------------------------------------------------------------------------
def output_size(n, k, p=0, s=1):
    """Размер выхода свёртки: ⌊(N + 2P − K) / S⌋ + 1."""
    return (n + 2 * p - k) // s + 1


def conv2d(x, kernel, bias=0.0, stride=1, padding=0):
    """Двумерная свёртка одного канала (как в CNN — без переворота ядра).

    x       — картинка H×W,
    kernel  — ядро K×K,
    stride  — шаг, с которым окно сдвигается,
    padding — ширина рамки из нулей вокруг картинки.
    out[i, j] = Σ_a Σ_b x_pad[i·S + a, j·S + b] · kernel[a, b] + bias
    """
    x = np.pad(x, padding)                  # рамка из нулей
    k = kernel.shape[0]
    h = (x.shape[0] - k) // stride + 1
    w = (x.shape[1] - k) // stride + 1
    out = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            window = x[i * stride:i * stride + k, j * stride:j * stride + k]
            out[i, j] = np.sum(window * kernel) + bias
    return out


def conv2d_multi(x, kernels, biases, stride=1, padding=0):
    """Свёртка C_in каналов в C_out карт: x C_in×H×W, kernels C_out×C_in×K×K.

    Каждый выходной канал — сумма свёрток всех входных каналов со «своими»
    срезами ядра плюс одно смещение.
    """
    return np.stack([
        sum(conv2d(x[c], kernels[o, c], stride=stride, padding=padding)
            for c in range(x.shape[0])) + biases[o]
        for o in range(kernels.shape[0])
    ])


def maxpool(x, size=2):
    """MaxPool: максимум в каждом неперекрывающемся квадрате size×size."""
    h, w = x.shape[0] // size, x.shape[1] // size
    return x[:h * size, :w * size].reshape(h, size, w, size).max(axis=(1, 3))


def avgpool(x, size=2):
    """AvgPool: среднее в каждом квадрате size×size."""
    h, w = x.shape[0] // size, x.shape[1] // size
    return x[:h * size, :w * size].reshape(h, size, w, size).mean(axis=(1, 3))


def check_against_torch():
    """Сравнить нашу реализацию с PyTorch на случайных данных."""
    section("1. Реализация свёртки и сверка с PyTorch")
    rng = np.random.default_rng(0)
    worst = 0.0
    for (n, k, p, s) in [(7, 3, 0, 1), (7, 3, 1, 1), (7, 3, 1, 2), (8, 5, 2, 1), (9, 3, 0, 3)]:
        x = rng.normal(size=(3, n, n))
        kern = rng.normal(size=(4, 3, k, k))
        b = rng.normal(size=4)
        ours = conv2d_multi(x, kern, b, stride=s, padding=p)
        ref = F.conv2d(torch.tensor(x)[None], torch.tensor(kern), torch.tensor(b),
                       stride=s, padding=p)[0].numpy()
        diff = np.abs(ours - ref).max()
        worst = max(worst, diff)
        print(f"  N={n}, K={k}, P={p}, S={s}: выход {ours.shape[1]}×{ours.shape[2]} "
              f"(формула: {output_size(n, k, p, s)}), max|разница| = {diff:.1e}")
    print(f"  Максимальное расхождение с PyTorch: {worst:.1e}")
    x = rng.normal(size=(6, 6))
    ref_mp = F.max_pool2d(torch.tensor(x)[None, None], 2)[0, 0].numpy()
    ref_ap = F.avg_pool2d(torch.tensor(x)[None, None], 2)[0, 0].numpy()
    print(f"  MaxPool / AvgPool, max|разница|: {np.abs(maxpool(x) - ref_mp).max():.1e} / "
          f"{np.abs(avgpool(x) - ref_ap).max():.1e}")
    return worst


# ---------------------------------------------------------------------------
# 2. Классические ядра
# ---------------------------------------------------------------------------
KERNELS = {
    "размытие": np.full((3, 3), 1 / 9),
    "Собель: вертикальные границы": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Собель: горизонтальные границы": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "Лаплас: контур": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
}


def plot_kernels(img):
    """Картинка и результаты её свёртки с четырьмя классическими ядрами."""
    section("2. Классические ядра как детекторы признаков")
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6),
                             gridspec_kw={"height_ratios": [1, 2.4]})
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img, cmap="gray_r")
    axes[1, 0].set_title("исходная\n28×28", fontsize=11)
    for col, (name, k) in enumerate(KERNELS.items(), start=1):
        fmt = "{:.2f}" if name == "размытие" else "{:+g}"
        draw_grid(axes[0, col], k, cmap=DIVERGING, vmin=-2, vmax=2, fmt=fmt, fontsize=10)
        out = conv2d(img, k, padding=1)
        lim = np.abs(out).max()
        cmap = "gray_r" if name == "размытие" else DIVERGING
        vmin = 0 if name == "размытие" else -lim
        axes[1, col].imshow(out, cmap=cmap, vmin=vmin, vmax=lim)
        axes[1, col].set_title(name.replace(": ", ":\n"), fontsize=11)
        print(f"  {name:<32} отклик от {out.min():+.2f} до {out.max():+.2f}")
    for ax in axes[1]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "kernels_on_image")


# ---------------------------------------------------------------------------
# Схемы: операция свёртки, padding/stride, пулинг
# ---------------------------------------------------------------------------
def plot_conv_operation():
    """Окно 3×3 на входе 5×5, ядро и выход 3×3 (одно значение посчитано)."""
    x = np.array([[1, 0, 2, 1, 0],
                  [0, 1, 3, 1, 0],
                  [0, 0, 2, 2, 1],
                  [1, 0, 1, 0, 0],
                  [2, 1, 0, 0, 1]], float)
    k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], float)
    out = conv2d(x, k)
    fig = plt.figure(figsize=(10, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 3, 3], wspace=0.45)
    a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_grid(a[0], x, "вход 5×5", highlight=[(1, 1, 3, 3)])
    draw_grid(a[1], k, "ядро 3×3", fmt="{:+g}", highlight=[(0, 0, 3, 3)])
    draw_grid(a[2], out, "выход 3×3", fmt="{:+g}", highlight=[(1, 1, 1, 1)])
    arrow_between(fig, a[0], a[1])
    arrow_between(fig, a[1], a[2])
    save(fig, "conv_operation")
    win = x[1:4, 1:4]
    terms = " + ".join(f"{int(v)}·({int(w):+d})" for v, w in zip(win.ravel(), k.ravel()))
    subsection("Схема «операция свёртки»")
    print(f"  out[1,1] = {terms} = {out[1, 1]:g}")
    print("  Выход:\n" + matrix_str(out, "{:4g}"))
    return x.tolist(), k.tolist(), out.tolist()


def _draw_padded(ax, n, p, k, s, title):
    """Вход n×n с рамкой p, первые два положения окна k×k с шагом s."""
    size = n + 2 * p
    ax.set_xlim(-0.2, size + 0.2)
    ax.set_ylim(size + 0.2, -0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(size):
        for j in range(size):
            pad = i < p or j < p or i >= n + p or j >= n + p
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#f4f5f7" if pad else "white",
                                   edgecolor="#c9ccd3", lw=1, ls="--" if pad else "-"))
            if pad:
                ax.text(j + 0.5, i + 0.5, "0", ha="center", va="center", fontsize=9,
                        color=MUTED)
    # первое положение окна — сплошная рамка, следующее (через stride) — пунктир
    ax.add_patch(Rectangle((0.08, 0.08), k - 0.16, k - 0.16, fill=False,
                           edgecolor=CNN_COLOR, lw=3.2))
    ax.add_patch(Rectangle((s + 0.2, 0.2), k - 0.4, k - 0.4, fill=False,
                           edgecolor=MLP_COLOR, lw=3, ls=(0, (4, 2))))
    out = output_size(n, k, p, s)
    ax.set_title(title, fontsize=12, color=INK)
    ax.text(size / 2, size + 0.6, f"выход {out}×{out}", ha="center", va="top",
            fontsize=12, color=INK_2)


def plot_padding_stride():
    """Три варианта: P=0,S=1; P=1,S=1; P=1,S=2 для входа 5×5 и ядра 3×3."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    _draw_padded(axes[0], 5, 0, 3, 1, "padding = 0, stride = 1")
    _draw_padded(axes[1], 5, 1, 3, 1, "padding = 1, stride = 1")
    _draw_padded(axes[2], 5, 1, 3, 2, "padding = 1, stride = 2")
    fig.text(0.5, -0.02, "сплошная рамка — первое положение окна 3×3, пунктир — следующее "
             "(сдвиг на stride); серые клетки — добавленные нули (padding)",
             ha="center", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, "padding_stride")


def plot_pooling():
    """MaxPool и AvgPool 2×2 на матрице 4×4."""
    x = np.array([[1, 3, 2, 0],
                  [5, 6, 1, 2],
                  [0, 2, 9, 4],
                  [1, 1, 3, 8]], float)
    fig = plt.figure(figsize=(9.5, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 2, 2], wspace=0.6)
    a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    colors = [CNN_COLOR, MLP_COLOR, TEAL, "#f2a541"]
    draw_grid(a[0], x, "вход 4×4")
    for (r, c), col in zip([(0, 0), (0, 2), (2, 0), (2, 2)], colors):
        a[0].add_patch(Rectangle((c + 0.05, r + 0.05), 1.9, 1.9, fill=False, edgecolor=col, lw=3))
    draw_grid(a[1], maxpool(x), "MaxPool 2×2")
    draw_grid(a[2], avgpool(x), "AvgPool 2×2", fmt="{:.2f}")
    for ax in a[1:]:
        for (r, c), col in zip([(0, 0), (0, 1), (1, 0), (1, 1)], colors):
            ax.add_patch(Rectangle((c + 0.05, r + 0.05), 0.9, 0.9, fill=False, edgecolor=col, lw=3))
    arrow_between(fig, a[0], a[1])
    save(fig, "pooling")
    return maxpool(x).tolist(), avgpool(x).tolist()


# ---------------------------------------------------------------------------
# Многоканальная свёртка и число параметров (раздел 3)
# ---------------------------------------------------------------------------
def _stack(ax, x0, y0, w, h, depth, n, color, label=None, dx=0.18, dy=0.14):
    """Нарисовать «стопку» из n плоскостей (псевдо-3D)."""
    for i in reversed(range(n)):
        ox, oy = x0 + i * dx * depth / max(n, 1), y0 + i * dy * depth / max(n, 1)
        ax.add_patch(Rectangle((ox, oy), w, h, facecolor=color, edgecolor="white", lw=1.2,
                               alpha=0.9))
    if label:
        ax.text(x0 + w / 2 + dx * depth / 2, y0 - 0.35, label, ha="center", va="top",
                fontsize=11, color=INK_2)


def plot_multichannel():
    """Схема: вход C_in×H×W, фильтр C_in×K×K, выход — одна карта на фильтр."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.2, 4.2)
    ax.axis("off")
    _stack(ax, 0.2, 0.3, 2.8, 2.8, 3, 3, "#c9d9e6", "вход: C_in × H × W\n(3 × 28 × 28)")
    ax.add_patch(Rectangle((0.9, 1.6), 0.9, 0.9, fill=False, edgecolor=CNN_COLOR, lw=2.5))
    _stack(ax, 4.6, 1.1, 1.0, 1.0, 3, 3, "#f6c3b2", "один фильтр:\nC_in × K × K (3 × 3 × 3)")
    ax.text(5.6, 2.9, "× C_out\nфильтров", ha="center", fontsize=11, color=CNN_COLOR)
    _stack(ax, 8.2, 0.5, 2.4, 2.4, 3.5, 6, "#f3a58d", "выход: C_out × H' × W'\n(6 × 28 × 28)")
    for xa, xb in ((3.9, 4.4), (7.0, 7.9)):
        ax.annotate("", xy=(xb, 1.9), xytext=(xa, 1.9),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(11.5, 3.9, "параметров:\nC_out·(C_in·K² + 1)\n= 6·(3·9 + 1) = 168",
            ha="left", va="top", fontsize=11, color=INK)
    save(fig, "multichannel_conv")


def param_comparison():
    """Число параметров: свёрточный слой против полносвязного."""
    section("3. Число параметров: свёрточный слой и полносвязный")
    rows = []
    cases = [
        ("Fashion-MNIST 1×28×28 → 32 карты 28×28", 1, 28, 32, 3),
        ("RGB 3×224×224 → 64 карты 224×224", 3, 224, 64, 3),
    ]
    for name, c_in, n, c_out, k in cases:
        conv = c_out * (c_in * k * k + 1)
        fc_same = (c_in * n * n) * (c_out * n * n) + c_out * n * n   # тот же размер выхода
        fc_1000 = (c_in * n * n) * 1000 + 1000                        # всего 1000 нейронов
        rows.append(dict(case=name, conv=conv, fc_same=fc_same, fc_1000=fc_1000))
        print(f"  {name}")
        print(f"    свёртка {k}×{k}:                       {fmt_int(conv):>16}")
        print(f"    полносвязный, тот же размер выхода:  {fmt_int(fc_same):>16}")
        print(f"    полносвязный, всего 1000 нейронов:   {fmt_int(fc_1000):>16}")
    return rows


# ---------------------------------------------------------------------------
# 4. Эквивариантность к сдвигу
# ---------------------------------------------------------------------------
def shift(img, dy, dx):
    """Сдвинуть картинку на (dy, dx) пикселей, освободившееся место — нули."""
    out = np.zeros_like(img)
    h, w = img.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def equivariance(img):
    """conv(shift(x)) == shift(conv(x)) — проверка и рисунок."""
    section("4. Эквивариантность свёртки к сдвигу")
    k = KERNELS["Собель: вертикальные границы"]
    dy, dx = 3, -4
    a = conv2d(shift(img, dy, dx), k, padding=1)
    b = shift(conv2d(img, k, padding=1), dy, dx)
    # у краёв картинки рамка из нулей вносит отличия — сравниваем внутреннюю часть
    inner = (slice(5, 23), slice(5, 23))
    diff = np.abs(a[inner] - b[inner]).max()
    print(f"  Сдвиг на ({dy}, {dx}) пикселей, ядро Собеля")
    print(f"  max |conv(shift(x)) − shift(conv(x))| во внутренней области: {diff:.1e}")

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
    lim = np.abs(conv2d(img, k, padding=1)).max()
    panels = [(img, "картинка x", "gray_r", 0), (conv2d(img, k, padding=1), "conv(x)", DIVERGING, -lim),
              (shift(img, dy, dx), "сдвинутая картинка", "gray_r", 0), (a, "conv(сдвинутой)", DIVERGING, -lim)]
    for ax, (m, title, cmap, vmin) in zip(axes.ravel(), panels):
        ax.imshow(m, cmap=cmap, vmin=vmin, vmax=lim if vmin else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor="#c9ccd3", lw=1))
    fig.tight_layout()
    save(fig, "equivariance")
    return diff


# ---------------------------------------------------------------------------
# 5. Рецептивное поле
# ---------------------------------------------------------------------------
def receptive_field(img):
    """Рецептивное поле после каждого слоя SimpleCNN.

    r — сколько пикселей входа (по стороне) «видит» один нейрон слоя,
    j — на сколько пикселей входа сдвигается окно при шаге на 1 нейрон.
    Для слоя с ядром k и шагом s: r ← r + (k − 1)·j,  j ← j·s.
    """
    section("5. Рецептивное поле SimpleCNN")
    layers = [("Conv 3×3", 3, 1), ("MaxPool 2×2", 2, 2), ("Conv 3×3", 3, 1), ("MaxPool 2×2", 2, 2)]
    r, j, rows = 1, 1, []
    for name, k, s in layers:
        r, j = r + (k - 1) * j, j * s
        rows.append((name, r, j))
        print(f"  {name:<12} рецептивное поле {r:>2}×{r:<2}  шаг {j}")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(img, cmap="gray_r", extent=(0, 28, 28, 0))
    ax.set_xticks(range(0, 29, 4)), ax.set_yticks(range(0, 29, 4))
    ax.grid(False)
    cx = cy = 14
    colors = [CNN_COLOR, "#f2a541", MLP_COLOR, TEAL]
    for (name, size, _), col in zip(rows, colors):
        ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size,
                               fill=False, edgecolor=col, lw=2.6, label=f"{name}: {size}×{size}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=10)
    ax.set_title("Какую область входа видит один нейрон", fontsize=11)
    save(fig, "receptive_field")
    return rows


def main():
    images, labels = raw_images(train=False)
    img = images[IMG_INDEX].astype(float) / 255.0

    worst = check_against_torch()
    plot_kernels(img)
    x5, k3, out3 = plot_conv_operation()
    plot_padding_stride()
    mp, ap = plot_pooling()
    plot_multichannel()
    params = param_comparison()
    eq = equivariance(img)
    rf = receptive_field(img)

    save_json({"max_diff_torch": worst, "conv_example": {"x": x5, "k": k3, "out": out3},
               "pooling": {"max": mp, "avg": ap}, "params": params,
               "equivariance_diff": eq, "receptive_field": rf}, "convolution_basics")


if __name__ == "__main__":
    main()

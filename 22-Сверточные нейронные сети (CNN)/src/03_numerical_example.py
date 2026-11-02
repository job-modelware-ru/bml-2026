"""Численный пример: один шаг обучения крошечной свёрточной сети вручную.

Сеть (все числа подобраны так, чтобы расчёт помещался на слайд):

    картинка X 6×6  →  свёртка с ядром K 3×3 (смещение b_K)  →  карта S 4×4
                    →  ReLU  →  R 4×4
                    →  MaxPool 2×2  →  P 2×2  →  вектор f длины 4
                    →  полносвязный слой z = W·f + b  (2 класса)
                    →  softmax  →  вероятности p  →  кросс-энтропия L

Класс 0 — «вертикальный штрих», класс 1 — «прочее». На картинке вертикальный
штрих, поэтому верный ответ y = 0. Ядро K — детектор вертикальной линии.

Скрипт:
    1) считает прямой проход по формулам (NumPy, явные циклы);
    2) считает градиенты обратным проходом вручную;
    3) делает шаг градиентного спуска и показывает, что потеря уменьшилась;
    4) сверяет все градиенты с автоматическим дифференцированием PyTorch.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (CNN_COLOR, INK, INK_2, MLP_COLOR, matrix_str, save,
                    save_json, section, subsection)
from diagrams import arrow_between, draw_grid

# ---------------------------------------------------------------------------
# Исходные данные
# ---------------------------------------------------------------------------
X = np.array([[0, 0, 1, 0, 0, 0],
              [0, 0, 2, 1, 0, 0],
              [0, 0, 2, 0, 0, 0],
              [0, 1, 2, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]], dtype=float)   # вертикальный штрих

K = np.array([[-1, 2, -1],
              [-1, 2, -1],
              [-1, 2, -1]], dtype=float)          # детектор вертикальной линии
B_K = 0.0                                          # смещение свёртки

W = np.array([[0.15, -0.10, 0.12, 0.05],
              [0.10, 0.10, 0.10, -0.05]])          # полносвязный слой 4 → 2
B = np.array([0.0, 0.3])
Y = 0                                              # верный класс
LR = 0.01                                          # шаг градиентного спуска


# ---------------------------------------------------------------------------
# Элементарные операции (так, как они записаны в формулах)
# ---------------------------------------------------------------------------
def conv2d(x, k, b=0.0):
    """Свёртка без отступов и с шагом 1: S[i, j] = Σ_a Σ_b X[i+a, j+b]·K[a, b] + b.

    Строго говоря, это взаимная корреляция (ядро не переворачивается);
    в глубоком обучении её принято называть свёрткой.
    """
    kh, kw = k.shape
    out = np.zeros((x.shape[0] - kh + 1, x.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k) + b
    return out


def maxpool2x2(r):
    """MaxPool 2×2 с шагом 2. Возвращает результат и позиции максимумов."""
    h, w = r.shape[0] // 2, r.shape[1] // 2
    out, where = np.zeros((h, w)), {}
    for i in range(h):
        for j in range(w):
            block = r[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            a, b = np.unravel_index(np.argmax(block), block.shape)
            out[i, j] = block[a, b]
            where[(i, j)] = (2 * i + a, 2 * j + b)
    return out, where


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def forward(k, b_k, w, b):
    """Прямой проход; возвращает все промежуточные величины."""
    s = conv2d(X, k, b_k)
    r = np.maximum(s, 0)
    p_map, where = maxpool2x2(r)
    f = p_map.ravel()
    z = w @ f + b
    p = softmax(z)
    loss = -np.log(p[Y])
    return dict(S=s, R=r, P=p_map, where=where, f=f, z=z, p=p, loss=loss)


def backward(fw, w):
    """Обратный проход: градиенты L по всем параметрам (правило цепочки)."""
    onehot = np.eye(2)[Y]
    dz = fw["p"] - onehot                    # ∂L/∂z = p − y   (softmax + CE)
    dW = np.outer(dz, fw["f"])               # ∂L/∂W = dz · fᵀ
    dB = dz                                  # ∂L/∂b = dz
    df = w.T @ dz                            # ∂L/∂f = Wᵀ · dz
    dP = df.reshape(2, 2)

    dR = np.zeros_like(fw["R"])              # MaxPool: градиент идёт только
    for (i, j), (a, b) in fw["where"].items():   # в позицию максимума
        dR[a, b] = dP[i, j]
    dS = dR * (fw["S"] > 0) + 0.0            # ReLU: пропускает там, где S > 0

    dK = conv2d(X, dS)                       # ∂L/∂K — свёртка входа с dS
    dBK = dS.sum()                           # ∂L/∂b_K — сумма dS
    return dict(dz=dz, dW=dW, dB=dB, df=df, dR=dR, dS=dS, dK=dK, dBK=dBK)


def torch_check():
    """Те же вычисления в PyTorch: autograd должен дать те же градиенты."""
    x = torch.tensor(X, dtype=torch.float64).view(1, 1, 6, 6)
    k = torch.tensor(K, dtype=torch.float64, requires_grad=True)
    bk = torch.tensor([B_K], dtype=torch.float64, requires_grad=True)
    w = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(B, dtype=torch.float64, requires_grad=True)

    s = F.conv2d(x, k.view(1, 1, 3, 3), bk)
    f = F.max_pool2d(F.relu(s), 2).flatten()
    z = w @ f + b
    loss = F.cross_entropy(z.view(1, -1), torch.tensor([Y]))
    loss.backward()
    return dict(S=s.detach().numpy()[0, 0], loss=loss.item(), dK=k.grad.numpy(),
                dBK=bk.grad.item(), dW=w.grad.numpy(), dB=b.grad.numpy())


# ---------------------------------------------------------------------------
# Рисунки
# ---------------------------------------------------------------------------
def plot_forward(fw):
    """Схема прямого прохода с числами."""
    fig = plt.figure(figsize=(14, 3.9))
    widths = [6, 3, 4, 4, 2, 2.4]
    gs = fig.add_gridspec(1, 6, width_ratios=widths, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    draw_grid(axes[0], X, "вход X (6×6)", highlight=[(0, 1, 3, 3)])
    draw_grid(axes[1], K, "ядро K (3×3)", fmt="{:+g}", highlight=[(0, 0, 3, 3)])
    draw_grid(axes[2], fw["S"], "свёртка S = X * K", highlight=[(0, 1, 1, 1)])
    draw_grid(axes[3], fw["R"], "ReLU: R = max(0, S)",
              highlight=[(0, 0, 2, 2), (0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 2, 2)],
              highlight_color=MLP_COLOR)
    draw_grid(axes[4], fw["P"], "MaxPool 2×2")
    ax = axes[5]
    ax.axis("off")
    ax.text(0.5, 0.86, "softmax", ha="center", va="center", fontsize=13,
            fontweight="bold", color=INK, transform=ax.transAxes)
    for c, (zc, pc) in enumerate(zip(fw["z"], fw["p"])):
        ax.text(0.5, 0.62 - 0.36 * c, f"класс {c}\nz = {zc:.2f}\np = {pc:.3f}",
                ha="center", va="center", fontsize=12,
                color=CNN_COLOR if c == Y else INK_2,
                transform=ax.transAxes)
    for a, b in zip(axes[1:-1], axes[2:]):
        arrow_between(fig, a, b)
    arrow_between(fig, axes[0], axes[1])
    save(fig, "numeric_forward")


def plot_backward(fw, bw, k_new, fw_new):
    """Обратный проход через свёртку: ∂L/∂K как сумма «градиент × окно входа».

    S[i,j] = Σ K[a,b]·X[i+a, j+b] + b_K, поэтому ∂S[i,j]/∂K — это окно X,
    которое ядро накрывало при вычислении S[i,j]. Ядро общее для всех позиций,
    поэтому вклады всех клеток складываются.
    """
    ds = bw["dS"]
    cells = [tuple(c) for c in np.argwhere(np.abs(ds) > 1e-12)]   # ненулевые клетки
    fig = plt.figure(figsize=(13, 4.13))
    n = len(cells)
    ratios = [4] + [3] * n + [3]
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.55,
                          left=0.02, right=0.98, top=0.86, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(ratios))]

    draw_grid(axes[0], ds, "∂L/∂S", fmt="{:.4f}", fontsize=10,
              highlight=[(r, c, 1, 1) for r, c in cells])
    for ax, (r, c) in zip(axes[1:1 + n], cells):
        draw_grid(ax, X[r:r + 3, c:c + 3], f"окно X под S[{r},{c}]", fontsize=11)
        ax.text(1.5, 3.25, f"× ({ds[r, c]:.4f})", ha="center", va="top",
                fontsize=12, color=CNN_COLOR, fontweight="bold")
    draw_grid(axes[-1], bw["dK"], "∂L/∂K", fmt="{:.3f}", fontsize=10)

    # знаки между панелями: → + … =
    signs = ["→"] + ["+"] * (n - 1) + ["="]
    for a, b, sgn in zip(axes[:-1], axes[1:], signs):
        if sgn == "→":
            arrow_between(fig, a, b)
        else:
            p1, p2 = a.get_position(), b.get_position()
            fig.text((p1.x1 + p2.x0) / 2, (p1.y0 + p1.y1) / 2, sgn, ha="center",
                     va="center", fontsize=22, color=INK)

    notes = ["≠ 0 только 2 клетки:\nMaxPool и ReLU обнулили остальные",
             "∂S[i,j]/∂K = окно X, которое\nядро накрывало для S[i,j]",
             "ядро общее → вклады\nвсех клеток складываются",
             f"∂L/∂b_K = Σ ∂L/∂S = {bw['dBK']:.4f}"]
    xs = [axes[0], axes[1], axes[n], axes[-1]]
    for ax, t in zip(xs, notes):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.07, t, ha="center", va="center",
                 fontsize=10.5, color=INK_2)
    save(fig, "numeric_backward")


# ---------------------------------------------------------------------------
def main():
    section("1. Прямой проход")
    fw = forward(K, B_K, W, B)

    subsection("Шаг 1. Свёртка: окно 3×3 скользит по X с шагом 1")
    win = X[0:3, 1:4]
    terms = " + ".join(f"({int(k):+d})·{int(x)}" for k, x in zip(K.ravel(), win.ravel()))
    print(f"  S[0,1] = {terms}\n         = {fw['S'][0, 1]:g}")
    print("  Карта S (4×4):\n" + matrix_str(fw["S"], "{:5g}"))
    print("  Размер выхода: (6 − 3) / 1 + 1 = 4")

    subsection("Шаг 2. ReLU и MaxPool 2×2")
    print("  R = max(0, S):\n" + matrix_str(fw["R"], "{:5g}"))
    print("  P = максимум в каждом квадрате 2×2:\n" + matrix_str(fw["P"], "{:5g}"))
    print(f"  f = flatten(P) = {fw['f']}")

    subsection("Шаг 3. Полносвязный слой, softmax, функция потерь")
    for c in range(2):
        terms = " + ".join(f"{w:+.2f}·{v:g}" for w, v in zip(W[c], fw["f"]))
        print(f"  z{c} = {terms} {B[c]:+.1f} = {fw['z'][c]:.2f}")
    print(f"  p = softmax(z) = [{fw['p'][0]:.4f}, {fw['p'][1]:.4f}]")
    print(f"  L = −ln p{Y} = −ln {fw['p'][Y]:.4f} = {fw['loss']:.4f}")

    section("2. Обратный проход (правило цепочки)")
    bw = backward(fw, W)
    print(f"  ∂L/∂z = p − y = [{bw['dz'][0]:+.4f}, {bw['dz'][1]:+.4f}]")
    print(f"  ∂L/∂f = Wᵀ·∂L/∂z = {np.round(bw['df'], 5)}")
    print("  ∂L/∂R — градиент только в позициях максимумов пулинга:\n"
          + matrix_str(bw["dR"], "{:8.4f}"))
    print("  ∂L/∂S — обнуляем там, где S ≤ 0 (ReLU):\n" + matrix_str(bw["dS"], "{:8.4f}"))
    print("  ∂L/∂K = свёртка X с ∂L/∂S:\n" + matrix_str(bw["dK"], "{:8.4f}"))
    print(f"  ∂L/∂b_K = Σ ∂L/∂S = {bw['dBK']:.4f}")
    print("  ∂L/∂W:\n" + matrix_str(bw["dW"], "{:8.4f}"))

    section(f"3. Шаг градиентного спуска, lr = {LR}")
    k_new = K - LR * bw["dK"]
    bk_new = B_K - LR * bw["dBK"]
    w_new = W - LR * bw["dW"]
    b_new = B - LR * bw["dB"]
    fw_new = forward(k_new, bk_new, w_new, b_new)
    print("  K после шага:\n" + matrix_str(k_new, "{:+8.4f}"))
    print(f"  p(класс 0): {fw['p'][0]:.4f} → {fw_new['p'][0]:.4f}")
    print(f"  L:          {fw['loss']:.4f} → {fw_new['loss']:.4f}")

    section("4. Сверка с PyTorch (autograd)")
    tc = torch_check()
    diffs = {
        "S": np.abs(tc["S"] - fw["S"]).max(),
        "L": abs(tc["loss"] - fw["loss"]),
        "∂L/∂K": np.abs(tc["dK"] - bw["dK"]).max(),
        "∂L/∂b_K": abs(tc["dBK"] - bw["dBK"]),
        "∂L/∂W": np.abs(tc["dW"] - bw["dW"]).max(),
        "∂L/∂b": np.abs(tc["dB"] - bw["dB"]).max(),
    }
    for name, d in diffs.items():
        print(f"  max |ручной − PyTorch| для {name:<8}: {d:.1e}")
    assert max(diffs.values()) < 1e-10, "ручной расчёт не совпал с PyTorch"
    print("  Все величины совпадают.")

    plot_forward(fw)
    plot_backward(fw, bw, k_new, fw_new)
    save_json({
        "X": X.tolist(), "K": K.tolist(), "W": W.tolist(), "B": B.tolist(), "lr": LR,
        "S": fw["S"].tolist(), "R": fw["R"].tolist(), "P": fw["P"].tolist(),
        "z": fw["z"].tolist(), "p": fw["p"].tolist(), "loss": fw["loss"],
        "dz": bw["dz"].tolist(), "df": bw["df"].tolist(), "dS": bw["dS"].tolist(),
        "dK": bw["dK"].tolist(), "dBK": bw["dBK"], "dW": bw["dW"].tolist(),
        "K_new": k_new.tolist(), "p_new": fw_new["p"].tolist(),
        "loss_new": fw_new["loss"], "max_diff_torch": max(diffs.values()),
    }, "numeric_example")


if __name__ == "__main__":
    main()

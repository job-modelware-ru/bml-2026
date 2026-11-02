"""Анализ обученной свёрточной сети: где ошибается и что выучила.

Использует веса из 04_train_cnn.py (если их нет — обучает модель заново).

1. Матрица ошибок и точность по классам.
2. Примеры ошибок.
3. Фильтры первого свёрточного слоя.
4. Карты признаков (выходы свёрточных блоков) для одной картинки.
5. Карта чувствительности к перекрытию (occlusion): какие области
   картинки важны для ответа.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CLASSES, INK, INK_2, RANDOM_STATE, get_device,
                    save, save_json, section)
from data import MEAN, STD, get_loaders
from engine import predict, run_experiment
from models import SimpleCNN

FEATURE_IMG = 0                 # картинка для карт признаков (ботинок)
OCCLUSION_CLASSES = (6, 4, 0)   # рубашка, пальто, футболка — классы, где сеть сомневается
PATCH, STEP = 6, 2              # размер и шаг серого квадрата для occlusion


def confusion(probs, labels):
    """Матрица ошибок: строка — истинный класс, столбец — предсказанный."""
    section("1. Матрица ошибок и точность по классам")
    pred = probs.argmax(1).numpy()
    y = labels.numpy()
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per_class = cm.diagonal() / cm.sum(1)
    for c in np.argsort(per_class):
        print(f"  {CLASSES[c]:<10} {per_class[c]:.3f}")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:5], off.shape))[0]
    print("\n  Самые частые ошибки (истина → прогноз):")
    for t, p in top:
        print(f"    {CLASSES[t]:<10} → {CLASSES[p]:<10} {off[t, p]:>4}")

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap="Reds", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(10), CLASSES, fontsize=10)
    ax.set_xlabel("предсказанный класс")
    ax.set_ylabel("истинный класс")
    ax.grid(False)
    ax.set_title("Матрица ошибок на тесте (10 000 картинок)", fontsize=12)
    fig.tight_layout()
    save(fig, "confusion_matrix")
    return cm, per_class, [(int(t), int(p), int(off[t, p])) for t, p in top]


def plot_errors(images, probs, labels, n=12):
    """Самые «уверенные» ошибки сети."""
    section("2. Примеры ошибок")
    pred = probs.argmax(1)
    wrong = torch.where(pred != labels)[0]
    conf = probs[wrong, pred[wrong]]
    order = wrong[torch.argsort(conf, descending=True)][:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(n // 2 * 1.8, 4.6))
    for ax, i in zip(axes.ravel(), order.tolist()):
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(f"{CLASSES[labels[i]]}\n→ {CLASSES[pred[i]]} ({probs[i, pred[i]]:.2f})",
                     fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle("истинный класс → ответ сети (уверенность)", fontsize=11, color=INK_2)
    fig.tight_layout()
    save(fig, "errors")
    print(f"  всего ошибок: {len(wrong)}; показаны {n} самых уверенных")
    return int(len(wrong))


def plot_filters(model):
    """32 ядра 3×3 первого свёрточного слоя."""
    section("3. Фильтры первого слоя")
    w = model.features[0].weight.detach().cpu().numpy()[:, 0]   # 32 × 3 × 3
    fig, axes = plt.subplots(4, 8, figsize=(8, 4.3))
    for k, ax in enumerate(axes.ravel()):
        lim = np.abs(w[k]).max()
        ax.imshow(w[k], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor("#c9ccd3")
    fig.suptitle("Веса 32 фильтров 3×3 первого слоя (красный > 0, синий < 0)",
                 fontsize=11, color=INK_2)
    fig.tight_layout()
    save(fig, "conv1_filters")
    print(f"  форма весов: {tuple(model.features[0].weight.shape)}")


@torch.no_grad()
def plot_feature_maps(model, image, device):
    """Карты признаков после 1-го и 2-го свёрточных блоков."""
    section("4. Карты признаков для одной картинки")
    x = image[None].to(device)
    after1 = model.features[:3](x)[0].cpu()     # Conv-BN-ReLU блока 1: 32×28×28
    after2 = model.features[:7](x)[0].cpu()     # … блока 2: 64×14×14
    # показываем 8 наиболее активных карт каждого блока
    top1 = after1.mean((1, 2)).argsort(descending=True)[:8]
    top2 = after2.mean((1, 2)).argsort(descending=True)[:8]
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(2, 9, width_ratios=[1.4] + [1] * 8, wspace=0.08, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(image[0] * STD + MEAN, cmap="gray_r")
    ax.text(14, -2, "вход", ha="center", va="bottom", fontsize=11, color=INK)
    ax.axis("off")
    for row, (maps, idx, title) in enumerate([(after1, top1, "блок 1: 28×28"),
                                              (after2, top2, "блок 2: 14×14")]):
        for k, c in enumerate(idx.tolist()):
            a = fig.add_subplot(gs[row, k + 1])
            a.imshow(maps[c], cmap="magma")
            a.axis("off")
            if k == 0:
                a.text(-0.08, 0.5, title, transform=a.transAxes, rotation=90,
                       ha="right", va="center", fontsize=10, color=INK_2)
            a.set_title(f"канал {c}", fontsize=8, color=INK_2)
    save(fig, "feature_maps")
    print(f"  блок 1: {tuple(after1.shape)}, блок 2: {tuple(after2.shape)}")


@torch.no_grad()
def occlusion_map(model, image, label, device):
    """Падение вероятности верного класса, когда квадрат PATCH×PATCH закрыт.

    Закрываем квадрат цветом фона (нулевая яркость) — как будто этой части
    вещи нет — и смотрим, насколько упала вероятность верного класса. Большое падение — область
    важна для решения сети.
    """
    model.eval()
    base = torch.softmax(model(image[None].to(device)), 1)[0, label].item()
    positions = range(0, 28 - PATCH + 1, STEP)
    batch, coords = [], []
    for i in positions:
        for j in positions:
            x = image.clone()
            x[:, i:i + PATCH, j:j + PATCH] = -MEAN / STD   # яркость 0 после нормализации
            batch.append(x)
            coords.append((i, j))
    p = torch.softmax(model(torch.stack(batch).to(device)), 1)[:, label].cpu()
    heat, count = np.zeros((28, 28)), np.zeros((28, 28))
    for (i, j), pij in zip(coords, p.tolist()):
        heat[i:i + PATCH, j:j + PATCH] += base - pij
        count[i:i + PATCH, j:j + PATCH] += 1
    return heat / np.maximum(count, 1), base


def pick_examples(probs, labels):
    """Для каждого класса из OCCLUSION_CLASSES — первая верно распознанная
    картинка с уверенностью 0.6…0.9 (у «уверенных» картинок вероятность
    почти не меняется, и карта получается пустой)."""
    p_true = probs[torch.arange(len(labels)), labels]
    ok = (probs.argmax(1) == labels) & (p_true > 0.6) & (p_true < 0.9)
    return [int(torch.where(ok & (labels == c))[0][0]) for c in OCCLUSION_CLASSES]


def plot_occlusion(model, test_ds, device, indices):
    section("5. Карта чувствительности к перекрытию (occlusion)")
    fig, axes = plt.subplots(2, len(indices), figsize=(3.2 * len(indices), 6.2))
    for k, idx in enumerate(indices):
        image, label = test_ds[idx]
        heat, base = occlusion_map(model, image, label, device)
        raw = image[0] * STD + MEAN
        axes[0, k].imshow(raw, cmap="gray_r")
        axes[0, k].set_title(f"{CLASSES[label]}: p = {base:.3f}", fontsize=11)
        axes[1, k].imshow(raw, cmap="gray_r")
        axes[1, k].imshow(heat, cmap="Reds", alpha=0.65, vmin=0, vmax=max(heat.max(), 1e-3))
        axes[1, k].set_title(f"макс. падение p: {heat.max():.2f}", fontsize=10, color=INK_2)
        for a in axes[:, k]:
            a.axis("off")
        print(f"  тест №{idx:<5} {CLASSES[label]:<9} p = {base:.3f}, "
              f"максимальное падение {heat.max():.3f}")
    fig.suptitle(f"Красное — области, закрытие которых квадратом {PATCH}×{PATCH} "
                 "сильнее всего снижает уверенность сети", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, "occlusion")


def main():
    device = get_device()
    loaders = get_loaders(seed=RANDOM_STATE)
    model, rec = run_experiment("SimpleCNN", SimpleCNN, loaders, device, seed=RANDOM_STATE)
    model.eval()
    test_ds = loaders[2].dataset
    probs, labels = predict(model, loaders[2], device)
    images = test_ds.data.numpy()

    cm, per_class, top = confusion(probs, labels)
    n_wrong = plot_errors(images, probs, labels)
    plot_filters(model)
    plot_feature_maps(model, test_ds[FEATURE_IMG][0], device)
    plot_occlusion(model, test_ds, device, pick_examples(probs, labels))

    save_json({"test_acc": float((probs.argmax(1) == labels).float().mean()),
               "per_class_acc": {CLASSES[c]: float(a) for c, a in enumerate(per_class)},
               "top_confusions": [{"true": CLASSES[t], "pred": CLASSES[p], "count": n}
                                  for t, p, n in top],
               "n_wrong": n_wrong, "confusion_matrix": cm.tolist()}, "cnn_analysis")
    print(f"\n  (точность на тесте: {rec['test_acc']:.4f})")


if __name__ == "__main__":
    main()

"""Чем CNN отличается от MLP по существу: три эксперимента.

Индуктивное смещение (inductive bias) — допущения о данных, заложенные в
архитектуру. У CNN их два: важны локальные соседства пикселей, и один и
тот же признак может встретиться в любом месте картинки.

1. Сдвиг: модели, обученные на обычных картинках, проверяются на тесте,
   сдвинутом на 1…6 пикселей. CNN должна терять меньше.
2. Перемешанные пиксели: все картинки (и обучение, и тест) проходят через
   одну и ту же случайную перестановку 784 пикселей. Для MLP ничего не
   меняется (порядок входов ему безразличен), а CNN теряет то, на чём
   основана, — соседство пикселей.
3. Мало данных: обучение на 250…10 000 картинках. Удачные допущения
   особенно полезны, когда данных мало.

Модели из 06_compare_models.py берутся с диска (если их нет — обучаются).
Запуск: python 07_inductive_bias.py   (≈ 35 мин на процессоре)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from common import (CNN_COLOR, INK, MLP_COLOR, RANDOM_STATE, RESULTS_DIR, TEAL,
                    get_device, save, save_json, section, set_seed)
from data import MEAN, STD, get_datasets, get_loaders
from engine import evaluate, run_experiment
from models import MLP, SimpleCNN, TinyCNN

SHIFTS = range(0, 7)
TRAIN_SIZES = (250, 500, 1000, 2000, 5000, 10000)
SIZE_SEEDS = (42, 1, 2)
SIZE_STEPS = 1000           # число шагов обучения для каждого размера выборки
BACKGROUND = -MEAN / STD    # значение «чёрного» (нулевого) пикселя после нормализации


# ---------------------------------------------------------------------------
# 1. Устойчивость к сдвигу
# ---------------------------------------------------------------------------
def shift_batch(x, dy, dx):
    """Сдвинуть батч N×1×28×28 на (dy, dx), освободившееся место — фон."""
    out = torch.full_like(x, BACKGROUND)
    h, w = x.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        x[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


@torch.no_grad()
def accuracy_shifted(model, loader, d, device):
    """Средняя точность по четырём направлениям сдвига на d пикселей."""
    model.eval()
    accs = []
    for dy, dx in ((0, d), (0, -d), (d, 0), (-d, 0)):
        correct = n = 0
        for x, y in loader:
            pred = model(shift_batch(x, dy, dx).to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            n += len(y)
        accs.append(correct / n)
        if d == 0:
            break
    return float(np.mean(accs))


def shift_experiment(device):
    section("1. Точность на сдвинутых картинках")
    loaders = get_loaders(seed=RANDOM_STATE)
    models = {
        "MLP": run_experiment("MLP", MLP, loaders, device, RANDOM_STATE, verbose=False)[0],
        "TinyCNN": run_experiment("TinyCNN", TinyCNN, loaders, device, RANDOM_STATE, verbose=False)[0],
        "SimpleCNN": run_experiment("SimpleCNN", SimpleCNN, loaders, device, RANDOM_STATE, verbose=False)[0],
    }
    res = {name: [] for name in models}
    print("  сдвиг   " + "".join(f"{n:>12}" for n in models))
    for d in SHIFTS:
        for name, m in models.items():
            res[name].append(accuracy_shifted(m, loaders[2], d, device))
        print(f"  {d:>3} px " + "".join(f"{res[n][-1]:>12.4f}" for n in models))

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    style = {"MLP": (MLP_COLOR, "-o"), "TinyCNN": (TEAL, "-s"), "SimpleCNN": (CNN_COLOR, "-o")}
    for name, accs in res.items():
        color, fmt = style[name]
        ax.plot(list(SHIFTS), [100 * a for a in accs], fmt, color=color, label=name)
        ax.text(SHIFTS[-1] + 0.15, 100 * accs[-1], f"{100 * accs[-1]:.1f} %", va="center",
                fontsize=10, color=color)
    ax.set(xlabel="сдвиг тестовой картинки, пикселей", ylabel="точность, %",
           title="Устойчивость к сдвигу (обучение без сдвигов)")
    ax.set_xlim(-0.3, SHIFTS[-1] + 1.1)
    ax.set_xticks(list(SHIFTS))
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "shift_robustness")
    return res


# ---------------------------------------------------------------------------
# 2. Перемешанные пиксели
# ---------------------------------------------------------------------------
class PermutePixels:
    """Преобразование: переставить пиксели по фиксированной перестановке."""

    def __init__(self, seed=RANDOM_STATE):
        self.perm = torch.from_numpy(np.random.RandomState(seed).permutation(784))

    def __call__(self, x):
        return x.reshape(-1)[self.perm].reshape(1, 28, 28)


def permutation_experiment(device):
    section("2. Перемешанные пиксели (одна перестановка для всех картинок)")
    permute = PermutePixels()
    t_perm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,)),
                                 permute])
    plain = get_loaders(seed=RANDOM_STATE)
    perm = get_loaders(seed=RANDOM_STATE, transform=t_perm)
    res = {}
    for name, make in (("MLP", MLP), ("SimpleCNN", SimpleCNN)):
        _, r_plain = run_experiment(name, make, plain, device, RANDOM_STATE, verbose=False)
        _, r_perm = run_experiment(f"{name}_perm", make, perm, device, RANDOM_STATE)
        res[name] = {"plain": r_plain["test_acc"], "permuted": r_perm["test_acc"]}
        print(f"  {name:<10} обычные: {r_plain['test_acc']:.4f}   "
              f"перемешанные: {r_perm['test_acc']:.4f}   "
              f"разница: {100 * (r_perm['test_acc'] - r_plain['test_acc']):+.2f} п.п.")

    # рисунок: пример перестановки и столбцы
    image = plain[2].dataset[0][0]
    fig = plt.figure(figsize=(11, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.6], wspace=0.3)
    for k, (m, title) in enumerate([(image, "обычная"), (permute(image), "перемешанная")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(m[0], cmap="gray_r")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    for k, (key, label, alpha) in enumerate([("plain", "обычные картинки", 1.0),
                                             ("permuted", "перемешанные пиксели", 0.45)]):
        vals = [100 * res[n][key] for n in ("MLP", "SimpleCNN")]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, alpha=alpha,
                      color=[MLP_COLOR, CNN_COLOR], label=label, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center",
                    fontsize=10, color=INK)
    ax.set_xticks(x, ["MLP", "SimpleCNN"])
    ax.set_ylim(80, 95)
    ax.set_ylabel("точность на тесте, %")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "permutation")
    return res


# ---------------------------------------------------------------------------
# 3. Зависимость от объёма обучающей выборки
# ---------------------------------------------------------------------------
def train_steps(model, dataset, device, steps, seed, batch_size=64, lr=1e-3):
    """Обучить модель фиксированное число шагов на небольшой выборке."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
                        generator=g, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)   # как в engine.fit
    loss_fn = nn.CrossEntropyLoss()
    model.to(device).train()
    done = 0
    while done < steps:
        for x, y in loader:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
            if done == steps:
                break
    return model


def data_size_experiment(device, full):
    section("3. Точность в зависимости от объёма обучающей выборки")
    cache_path = RESULTS_DIR / "data_size_runs.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    train, _, test = get_datasets()
    test_loader = DataLoader(test, batch_size=1024)
    for n in TRAIN_SIZES:
        for name, make in (("MLP", MLP), ("SimpleCNN", SimpleCNN)):
            for seed in SIZE_SEEDS:
                key = f"{name}|{n}|{seed}"
                if key in cache:
                    continue
                set_seed(seed)
                idx = np.random.RandomState(seed).choice(len(train), n, replace=False)
                model = train_steps(make(), Subset(train, idx.tolist()), device, SIZE_STEPS, seed)
                cache[key] = evaluate(model, test_loader, device)[1]
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    res = {}
    print(f"  {'картинок':>9}{'MLP':>16}{'SimpleCNN':>16}{'разница':>10}")
    for n in TRAIN_SIZES:
        row = {}
        for name in ("MLP", "SimpleCNN"):
            v = np.array([cache[f"{name}|{n}|{s}"] for s in SIZE_SEEDS])
            row[name] = (float(v.mean()), float(v.std(ddof=1)))
        res[n] = row
        print(f"  {n:>9}{row['MLP'][0]:>10.4f} ±{row['MLP'][1]:.3f}"
              f"{row['SimpleCNN'][0]:>10.4f} ±{row['SimpleCNN'][1]:.3f}"
              f"{100 * (row['SimpleCNN'][0] - row['MLP'][0]):>+9.2f}")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    sizes = list(TRAIN_SIZES) + [55000]
    for name, color in (("MLP", MLP_COLOR), ("SimpleCNN", CNN_COLOR)):
        m = [100 * res[n][name][0] for n in TRAIN_SIZES] + [100 * full[name]]
        s = [100 * res[n][name][1] for n in TRAIN_SIZES] + [0]
        ax.errorbar(sizes, m, yerr=s, fmt="-o", color=color, label=name, capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{n:,}".replace(",", " ") for n in sizes], fontsize=9)
    ax.minorticks_off()
    ax.set(xlabel="размер обучающей выборки (лог. шкала)", ylabel="точность на тесте, %",
           title="Сколько нужно данных")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "data_size")
    return {str(n): v for n, v in res.items()}


def main():
    device = get_device()
    shift = shift_experiment(device)
    perm = permutation_experiment(device)
    full = {"MLP": perm["MLP"]["plain"], "SimpleCNN": perm["SimpleCNN"]["plain"]}
    sizes = data_size_experiment(device, full)
    save_json({"shift": {k: v for k, v in shift.items()}, "shifts": list(SHIFTS),
               "permutation": perm, "data_size": sizes, "data_size_steps": SIZE_STEPS},
              "inductive_bias")


if __name__ == "__main__":
    main()

"""Основной пример: обучение свёрточной сети на Fashion-MNIST.

Порядок работы:
    1. данные   — 55 000 / 5 000 / 10 000 картинок 28×28 (обучение / валидация / тест);
    2. модель   — SimpleCNN из models.py (два свёрточных блока + классификатор);
    3. обучение — 12 эпох, Adam, lr = 0.001 с косинусным уменьшением,
                  мини-батч 128, кросс-энтропия;
    4. оценка   — точность на тесте у версии модели с лучшей эпохой на валидации.

Результаты: results/models/SimpleCNN_seed42.pt (веса), results/cnn_training.json,
figures/cnn_learning_curves.png. На процессоре обучение занимает ~7 минут,
на видеокарте — меньше минуты.

Запуск:  python 04_train_cnn.py
"""

import matplotlib.pyplot as plt

from common import (CNN_COLOR, INK_2, MUTED, RANDOM_STATE, get_device, save,
                    save_json, section)
from data import get_loaders
from engine import run_experiment
from models import SimpleCNN, count_params, fmt_int, layer_table

EPOCHS = 12
LR = 1e-3
BATCH_SIZE = 128


def describe_data(loaders):
    """Показать размеры выборок и форму одного мини-батча."""
    section("1. Данные: Fashion-MNIST")
    train_loader, val_loader, test_loader = loaders
    for name, loader in (("обучение", train_loader), ("валидация", val_loader),
                         ("тест", test_loader)):
        print(f"  {name:<10} {len(loader.dataset):>6} картинок")
    images, labels = next(iter(train_loader))
    print(f"\n  Один мини-батч: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print("  (N × C × H × W: 128 картинок × 1 канал × 28 × 28 пикселей)")
    print(f"  Яркость после нормализации: среднее {images.mean():+.3f}, "
          f"std {images.std():.3f}")


def describe_model(model):
    """Таблица слоёв: форма выхода и число параметров."""
    section("2. Модель: SimpleCNN")
    total = count_params(model)
    print(f"  {'слой':<14}{'выход':<16}{'параметров':>11}")
    for name, shape, n in layer_table(model):
        print(f"  {name:<14}{' × '.join(map(str, shape)):<16}{fmt_int(n):>11}")
    conv = sum(p.numel() for p in model.features.parameters())
    print(f"\n  Всего параметров: {fmt_int(total)}")
    print(f"  из них в свёрточной части: {fmt_int(conv)} ({conv / total:.1%}), "
          f"в полносвязной: {fmt_int(total - conv)} ({1 - conv / total:.1%})")
    return total, conv


def plot_curves(history, test_acc):
    """Кривые обучения: функция потерь и точность по эпохам."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history["best_epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.plot(epochs, history["train_loss"], "-o", color=CNN_COLOR, label="обучение")
    ax1.plot(epochs, history["val_loss"], "--o", color=INK_2, label="валидация")
    ax1.set(title="Функция потерь (кросс-энтропия)", xlabel="эпоха", ylabel="loss")

    ax2.plot(epochs, [100 * a for a in history["train_acc"]], "-o", color=CNN_COLOR,
             label="обучение")
    ax2.plot(epochs, [100 * a for a in history["val_acc"]], "--o", color=INK_2,
             label="валидация")
    ax2.axhline(100 * test_acc, color=MUTED, lw=1, ls=":")
    ax2.text(1, 100 * test_acc + 0.15, f"тест: {100 * test_acc:.2f} %", color=MUTED,
             fontsize=10, va="bottom")
    ax2.set(title="Точность (accuracy), %", xlabel="эпоха", ylabel="%")

    for ax in (ax1, ax2):
        ax.axvline(best, color=MUTED, lw=1, ls="--")
        ax.set_xticks(list(epochs))
        ax.legend()
    ax1.text(best - 0.15, ax1.get_ylim()[1] * 0.97, "лучшая эпоха", color=MUTED,
             fontsize=9, va="top", ha="right")
    fig.tight_layout()
    save(fig, "cnn_learning_curves")


def main():
    device = get_device()
    print(f"Устройство: {device}")
    loaders = get_loaders(batch_size=BATCH_SIZE, seed=RANDOM_STATE)
    describe_data(loaders)

    total, conv = describe_model(SimpleCNN())

    section("3. Обучение")
    print(f"  {EPOCHS} эпох, Adam, lr = {LR} → 0 (по косинусу), batch = {BATCH_SIZE}")
    print("  колонки: loss обучение / валидация, accuracy обучение / валидация\n")
    model, rec = run_experiment("SimpleCNN", SimpleCNN, loaders, device,
                                seed=RANDOM_STATE, epochs=EPOCHS, lr=LR)

    section("4. Итог")
    print(f"  лучшая эпоха по валидации: {rec['best_epoch']} "
          f"(val acc = {rec['best_val_acc']:.4f})")
    print(f"  точность на тесте:         {rec['test_acc']:.4f} "
          f"({round(rec['test_acc'] * 10_000)} из 10 000)")
    print(f"  время обучения:            {rec['train_time']:.0f} с "
          f"({rec['train_time'] / EPOCHS:.1f} с на эпоху)")
    gap = rec["train_acc"][-1] - rec["val_acc"][-1]
    print(f"  разрыв train − val в последней эпохе: {gap * 100:.2f} п.п.")

    plot_curves(rec, rec["test_acc"])
    save_json({
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        "n_params": total, "n_params_conv": conv,
        "best_epoch": rec["best_epoch"], "best_val_acc": rec["best_val_acc"],
        "test_acc": rec["test_acc"], "test_loss": rec["test_loss"],
        "train_time": rec["train_time"],
        "train_acc": rec["train_acc"], "val_acc": rec["val_acc"],
        "train_loss": rec["train_loss"], "val_loss": rec["val_loss"],
    }, "cnn_training")


if __name__ == "__main__":
    main()

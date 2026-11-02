"""Цикл обучения и оценки — общий для CNN и MLP.

Одна эпоха = один проход по всей обучающей выборке. На каждом мини-батче:
    1) прямой проход: модель выдаёт логиты (оценки классов);
    2) функция потерь: кросс-энтропия между softmax(логиты) и верной меткой;
    3) обратный проход: loss.backward() считает градиенты по всем весам
       (метод обратного распространения ошибки);
    4) шаг оптимизатора: веса сдвигаются против градиента.
После каждой эпохи шаг обучения уменьшается по косинусу (cosine annealing):
в начале сеть учится быстро, к концу — аккуратно «доводит» веса.
"""

import copy
import json
import time

import torch
from torch import nn

from common import RESULTS_DIR, set_seed


def train_one_epoch(model, loader, optimizer, device):
    """Одна эпоха обучения. Возвращает средние loss и accuracy на обучении."""
    model.train()                     # включает Dropout и BatchNorm в режим обучения
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)                 # 1) прямой проход
        loss = loss_fn(logits, labels)         # 2) функция потерь

        optimizer.zero_grad()                  # обнуляем градиенты прошлого шага
        loss.backward()                        # 3) обратный проход
        optimizer.step()                       # 4) обновление весов

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(labels)

    return total_loss / n, correct / n


@torch.no_grad()                                # градиенты при оценке не нужны
def evaluate(model, loader, device):
    """Средние loss и accuracy на выборке (без обучения)."""
    model.eval()                      # Dropout выключен, BatchNorm — по накопленной статистике
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    """Вероятности классов и верные метки для всей выборки."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def fit(model, train_loader, val_loader, device, epochs=12, lr=1e-3, verbose=True):
    """Обучить модель и вернуть лучшую по валидации версию и историю.

    Оптимизатор Adam — градиентный спуск, который подбирает свой шаг для
    каждого веса. Начальный шаг lr плавно уменьшается до нуля к последней
    эпохе (CosineAnnealingLR): большие шаги в конце обучения раскачивают
    веса, и точность на валидации «скачет» от эпохи к эпохе.
    Если точность на валидации перестаёт расти, мы всё равно доучиваем до
    конца, но запоминаем веса лучшей эпохи («ранняя остановка» без досрочного выхода).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": []}
    best_acc, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()                                   # уменьшить шаг обучения
        va_loss, va_acc = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0

        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc, dt)):
            history[key].append(value)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            mark = "  ← лучшая" if epoch == best_epoch else ""
            print(f"  эпоха {epoch:2d}: loss {tr_loss:.4f} / {va_loss:.4f}   "
                  f"acc {tr_acc:.4f} / {va_acc:.4f}   {dt:5.1f} с{mark}")

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_acc
    return model, history


# ---------------------------------------------------------------------------
# Запуск с сохранением: обученная модель не переобучается повторно
# ---------------------------------------------------------------------------
MODELS_DIR = RESULTS_DIR / "models"


def run_experiment(name, make_model, loaders, device, seed, epochs=12, lr=1e-3,
                   force=False, verbose=True):
    """Обучить make_model() с данным seed или загрузить готовый результат.

    Веса сохраняются в results/models/<name>_seed<seed>.pt, числа — рядом
    в .json. Повторный запуск любого скрипта берёт их с диска, поэтому
    модели, общие для нескольких скриптов, обучаются один раз.
    Возвращает (модель, словарь с историей и итоговыми метриками).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = MODELS_DIR / f"{name}_seed{seed}.pt"
    record_path = weights.with_suffix(".json")
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    model = make_model()
    if weights.exists() and record_path.exists() and not force:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.to(device)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if verbose:
            print(f"  {name}, seed={seed}: загружено с диска "
                  f"(test acc = {record['test_acc']:.4f})")
        return model, record

    if verbose:
        print(f"  {name}, seed={seed}: обучение {epochs} эпох")
    t0 = time.perf_counter()
    model, history = fit(model, train_loader, val_loader, device,
                         epochs=epochs, lr=lr, verbose=verbose)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_loss, test_acc = evaluate(model, test_loader, device)
    infer_time = time.perf_counter() - t0

    record = dict(history, name=name, seed=seed, epochs=epochs, lr=lr,
                  test_loss=test_loss, test_acc=test_acc,
                  train_time=train_time, infer_time_10k=infer_time,
                  n_params=sum(p.numel() for p in model.parameters()))
    torch.save(model.state_dict(), weights)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if verbose:
        print(f"  → test acc = {test_acc:.4f}, обучение {train_time:.0f} с")
    return model, record

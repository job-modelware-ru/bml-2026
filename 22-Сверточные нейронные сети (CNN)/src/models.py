"""Две модели, которые сравниваются в презентации.

* SimpleCNN — небольшая свёрточная сеть (два свёрточных блока + классификатор);
* MLP       — полносвязная сеть (многослойный перцептрон) сопоставимого размера.

Обозначения форм тензоров: N × C × H × W
    N — число картинок в мини-батче,
    C — число каналов (карт признаков),
    H, W — высота и ширина.
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Свёрточная сеть для картинок 1×28×28 и 10 классов.

    Часть features извлекает признаки, часть classifier по ним решает,
    какой это класс. Формы тензоров для одной картинки:

        вход                                   1 × 28 × 28
        Conv 3×3, 32 фильтра, padding=1   →   32 × 28 × 28
        BatchNorm, ReLU                   →   32 × 28 × 28
        MaxPool 2×2                       →   32 × 14 × 14
        Conv 3×3, 64 фильтра, padding=1   →   64 × 14 × 14
        BatchNorm, ReLU                   →   64 × 14 × 14
        MaxPool 2×2                       →   64 ×  7 ×  7
        Flatten                           →   3136
        Linear + ReLU + Dropout           →   128
        Linear                            →   10 (логиты классов)
    """

    def __init__(self, n_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Блок 1. Свёртка: 32 фильтра 3×3 скользят по картинке.
            # padding=1 добавляет рамку из нулей, чтобы размер не уменьшался.
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # BatchNorm нормирует каждую карту признаков по мини-батчу
            # (среднее 0, дисперсия 1) — обучение идёт быстрее и стабильнее.
            nn.BatchNorm2d(32),
            # ReLU(x) = max(0, x) — нелинейность; без неё стопка свёрток
            # схлопнулась бы в одну линейную операцию.
            nn.ReLU(),
            # MaxPool 2×2: из каждого квадрата 2×2 берём максимум —
            # размер уменьшается вдвое, сильный отклик сохраняется.
            nn.MaxPool2d(kernel_size=2),

            # Блок 2. Те же операции, но на вход уже 32 карты признаков,
            # и каждый из 64 фильтров имеет размер 32×3×3.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 64×7×7 → вектор длины 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            # Dropout при обучении случайно обнуляет 30 % нейронов —
            # сеть не может полагаться на отдельные нейроны (регуляризация).
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),    # логиты: softmax применит функция потерь
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """Полносвязная сеть: картинка вытягивается в вектор из 784 чисел.

        вход 1×28×28 → Flatten → 784
        Linear + ReLU + Dropout → 256
        Linear + ReLU + Dropout → 128
        Linear                  → 10

    О том, что пиксели образуют двумерную сетку, сеть не знает:
    соседние и далёкие пиксели для неё равноправны.
    """

    def __init__(self, n_classes=10, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, width = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    """Компактная свёрточная сеть с глобальным усреднением (GAP).

    В SimpleCNN 95 % параметров сидят в первом полносвязном слое
    (3136 × 128). Здесь его нет: после трёх свёрточных блоков каждая из
    128 карт признаков усредняется в одно число (Global Average Pooling),
    и классификатор — один слой 128 → 10. Так устроены ResNet, MobileNet и др.

        вход                          1 × 28 × 28
        Conv 3×3 (32) + BN + ReLU + MaxPool  →   32 × 14 × 14
        Conv 3×3 (64) + BN + ReLU + MaxPool  →   64 ×  7 ×  7
        Conv 3×3 (128) + BN + ReLU           →  128 ×  7 ×  7
        AdaptiveAvgPool → Flatten            →  128
        Linear                               →  10
    """

    def __init__(self, n_classes=10):
        super().__init__()

        def block(c_in, c_out, pool):
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(*block(1, 32, True), *block(32, 64, True),
                                      *block(64, 128, False))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # среднее по каждой карте 7×7 → 128×1×1
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    """Число обучаемых параметров (весов и смещений) модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model, input_shape=(1, 1, 28, 28)):
    """Таблица «слой — форма выхода — число параметров».

    Прогоняем через сеть одну пустую картинку и записываем форму выхода
    каждого слоя. Аналог model.summary() из Keras.
    """
    rows, x = [], torch.zeros(input_shape)
    model.eval()
    with torch.no_grad():
        for block in (model.features, model.classifier) if hasattr(model, "features") else (model.net,):
            for layer in block:
                x = layer(x)
                n = sum(p.numel() for p in layer.parameters())
                rows.append((layer.__class__.__name__, tuple(x.shape[1:]), n))
    return rows


def fmt_int(n):
    """421834 → «421 834» (пробел как разделитель разрядов)."""
    return f"{n:,}".replace(",", " ")


if __name__ == "__main__":
    for m in (SimpleCNN(), TinyCNN(), MLP()):
        print(f"\n{m.__class__.__name__}: {fmt_int(count_params(m))} параметров")
        for name, shape, n in layer_table(m):
            shape_str = " × ".join(map(str, shape))
            print(f"  {name:<18} {shape_str:<14} {fmt_int(n):>9}")

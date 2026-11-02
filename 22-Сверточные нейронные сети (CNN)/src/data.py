"""Загрузка Fashion-MNIST.

Fashion-MNIST (Zalando, 2017) — 70 000 чёрно-белых изображений одежды
размером 28×28 пикселей, 10 классов, 60 000 для обучения и 10 000 для
теста. По формату совпадает с классическим MNIST (рукописные цифры), но
заметно сложнее: на MNIST даже простые модели дают 97–98 %, и разница
между методами почти не видна.

При первом запуске torchvision скачивает архивы (~30 МБ) в src/data/.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import DATA_DIR, RANDOM_STATE

# Среднее и стандартное отклонение яркости пикселя по обучающей выборке
# (после деления на 255). Посчитаны один раз функцией pixel_stats() ниже.
MEAN, STD = 0.2860, 0.3530

# ToTensor: картинка uint8 [0, 255] → тензор float [0, 1] формы 1×28×28
#           (1 — число каналов: изображение чёрно-белое).
# Normalize: (x − MEAN) / STD — пиксели становятся около нуля с разбросом ~1;
#           так градиентному спуску проще подобрать веса.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])

N_VAL = 5_000  # сколько обучающих картинок отложить для валидации


def get_datasets(transform=TRANSFORM):
    """Обучающая (55 000), валидационная (5 000) и тестовая (10 000) выборки.

    Валидационная выборка нужна, чтобы следить за переобучением и выбрать
    лучшую эпоху, не подглядывая в тест. Тест используется один раз —
    для итоговой оценки.
    """
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                                       transform=transform)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True,
                                 transform=transform)

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(full_train))
    train = Subset(full_train, idx[N_VAL:].tolist())
    val = Subset(full_train, idx[:N_VAL].tolist())
    return train, val, test


def get_loaders(batch_size=128, transform=TRANSFORM, seed=RANDOM_STATE):
    """Загрузчики мини-батчей для трёх выборок.

    Мини-батч — порция из batch_size картинок, по которой считается один
    шаг градиентного спуска. Обучающая выборка перемешивается каждую эпоху
    (порядок задаётся seed, поэтому запуск воспроизводим).
    """
    train, val, test = get_datasets(transform)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(val, batch_size=1024),
            DataLoader(test, batch_size=1024))


def as_numpy():
    """Те же выборки в виде массивов NumPy (для моделей scikit-learn).

    Возвращает X_train (55 000×784), y_train, X_val, y_val, X_test, y_test;
    яркости в диапазоне [0, 1]. Каждая картинка вытянута в вектор длины 784.
    """
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    X_all = full_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_all = full_train.targets.numpy()
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(y_all))
    tr, va = idx[N_VAL:], idx[:N_VAL]

    X_test = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_all[tr], y_all[tr], X_all[va], y_all[va], X_test, test.targets.numpy()


def raw_images(train=True):
    """Исходные картинки uint8 формы N×28×28 и метки — для рисунков."""
    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()


def pixel_stats():
    """Посчитать MEAN и STD по обучающей выборке (константы выше)."""
    x = datasets.FashionMNIST(DATA_DIR, train=True, download=True).data.float() / 255
    return x.mean().item(), x.std().item()

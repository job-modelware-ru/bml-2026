"""Данные «ядовитая лягушка (+1) / безвредная лягушка (-1)».

"""
import numpy as np

FEATURES = ["яркость окраски (0-10)", "длина тела, мм", "доля активности днём",
            "влажность кожи (0-1)", "дальность прыжка, см", "высота обитания над землёй, м",
            "частота зова, кГц", "масса, г (слабый признак)", "температура воздуха, °C (шум)",
            "высота над уровнем моря, м (шум)", "час наблюдения (шум)", "номер водоёма (шум)"]

# (среднее ядовитых, среднее безвредных, std ядовитых, std безвредных)
_SPEC = [(7.0, 3.0, 2.0, 2.0), (35, 65, 12, 14), (0.75, 0.35, 0.20, 0.25), (0.80, 0.60, 0.12, 0.18),
         (40, 85, 15, 25), (3.0, 1.0, 2.0, 1.5), (3.2, 2.4, 0.6, 0.7), (9, 11, 4, 4)]


def make_frogs(n=4000, sep=1.0, flip=0.0, seed=0):
    """y = 1 — ядовитая лягушка, 0 — безвредная. sep < 1 сближает классы,
    flip — доля перевёрнутых меток."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    cols = []
    for mp, ms, sp, ss in _SPEC:
        mid = (mp + ms) / 2
        mp, ms = mid + (mp - mid) * sep, mid + (ms - mid) * sep
        cols.append(np.where(y == 1, rng.normal(mp, sp, n), rng.normal(ms, ss, n)))
    noise = [rng.normal(22, 4, n), rng.normal(300, 120, n), rng.uniform(0, 24, n), rng.integers(1, 20, n)]
    X = np.column_stack(cols + noise).astype(float)
    if flip:
        idx = rng.random(n) < flip
        y[idx] = 1 - y[idx]
    return X, y

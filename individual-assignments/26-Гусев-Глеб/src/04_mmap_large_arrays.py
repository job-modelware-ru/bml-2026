"""Пример 4. mmap_mode: загрузка больших numpy-массивов без чтения в память.

joblib хранит numpy-массивы отдельно от остального объекта, поэтому несжатый
файл можно «отобразить в память» (memory map): load почти ничего не читает,
а данные подгружаются с диска в момент обращения к ним.

Честное сравнение — «load + обращение к данным», поэтому меряем оба шага.
Каждый вариант запускается в отдельном процессе, чтобы замеры не влияли друг на друга.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PATH = Path(__file__).parent / "models" / "big_array.joblib"


def child(mode):
    t0 = time.perf_counter()
    data = joblib.load(PATH, mmap_mode="r" if mode == "mmap" else None)
    t1 = time.perf_counter()
    row = np.array(data["embeddings"][123])        # обращение к одной строке
    t2 = time.perf_counter()
    total = float(data["embeddings"].sum())        # обращение ко всему массиву
    t3 = time.perf_counter()
    print(json.dumps({"mode": mode, "type": type(data["embeddings"]).__name__,
                      "load_ms": (t1 - t0) * 1000, "row_ms": (t2 - t1) * 1000,
                      "all_ms": (t3 - t2) * 1000, "row3": row[:3].round(3).tolist()}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
        sys.exit()

    PATH.parent.mkdir(exist_ok=True)
    # Например, матрица эмбеддингов 20 000 x 2 000 float64 ≈ 305 МБ
    rng = np.random.default_rng(0)
    joblib.dump({"embeddings": rng.random((20_000, 2_000))}, PATH)  # без compress!
    print(f"Файл: {PATH.stat().st_size / 2**20:.0f} МБ")

    for mode in ["full", "mmap"]:
        out = subprocess.run([sys.executable, __file__, mode], capture_output=True,
                             text=True, check=True).stdout
        r = json.loads(out)
        name = "обычный load " if mode == "full" else "mmap_mode='r'"
        print(f"{name}: load {r['load_ms']:7.1f} мс + строка {r['row_ms']:5.2f} мс "
              f"= {r['load_ms'] + r['row_ms']:7.1f} мс | весь массив ещё "
              f"{r['all_ms']:5.0f} мс | тип {r['type']} | {r['row3']}")
    PATH.unlink()  # удаляем большой файл, чтобы не занимать место

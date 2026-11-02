"""Полный набор замеров для презентации. Результаты -> папка results/.

Части:
  1. compression — все методы сжатия x уровни 1..9 на случайном лесе (~80 МБ)
  2. mmap        — обычный load vs mmap_mode="r": load + обращение к строке
  3. formats     — pickle (протоколы 4 и 5) vs joblib vs joblib+mmap
                   на трёх масштабах: ~1 МБ, ~80 МБ, ~0.5-1 ГБ; время и память

Каждая загрузка в частях 2-3 выполняется в отдельном процессе: так замеры
не влияют друг на друга, а sklearn импортируется ДО старта таймера.

Запуск:  python bench_all.py            (всё)
         python bench_all.py formats    (только одна часть)
"""
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
TMP = HERE / "models" / "bench_tmp"


# утилиты
def quiet_lz4():
    """lz4 + joblib 1.6 печатают безвредное 'I/O operation on closed file' — глушим."""
    def hook(u):
        if "closed file" not in str(u.exc_value):
            sys.__unraisablehook__(u)
    sys.unraisablehook = hook


def best_of(fn, n):
    res = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        res.append(time.perf_counter() - t0)
    return min(res)


def adaptive(fn, max_n=3, slow=3.0):
    """Один прогон; если он быстрый (< slow c) — ещё до max_n и берём минимум."""
    t = best_of(fn, 1)
    if t < slow and max_n > 1:
        t = min(t, best_of(fn, max_n - 1))
    return t


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def log(*a):
    print(*a, flush=True)
    with open(RES / "log.txt", "a", encoding="utf-8") as f:
        print(*a, file=f)


def mem_now_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def mem_peak_mb():
    """Пиковое потребление памяти процессом за всё время жизни, МБ."""
    import psutil
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):                      # Windows
        return mi.peak_wset / 2**20
    if sys.platform.startswith("linux"):              # Linux: пик с момента exec
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    import resource                                   # macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def run_child(*args):
    out = subprocess.run([sys.executable, __file__, "--child", *args],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def system_info():
    import joblib, numpy, sklearn, psutil
    info = {
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_logical": os.cpu_count(), "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "ram_available_gb": round(psutil.virtual_memory().available / 2**30, 1),
        "disk_free_gb": round(psutil.disk_usage(str(HERE)).free / 2**30, 1),
        "python": sys.version.split()[0], "numpy": numpy.__version__,
        "sklearn": sklearn.__version__, "joblib": joblib.__version__,
    }
    try:
        import lz4
        info["lz4"] = lz4.__version__
    except ImportError:
        info["lz4"] = None
    return info


# модели
def model_rf_small():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_breast_cancer(return_X_y=True)
    return RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)


def model_rf_big():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=100_000, n_features=40, random_state=0)
    return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X, y)


def model_knn(n_rows):
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(0)
    X = rng.random((n_rows, 100))
    return KNeighborsClassifier().fit(X, rng.integers(0, 2, n_rows))


# дочерние процессы
def child(args):
    kind = args[0]
    import numpy as np
    import joblib
    import sklearn.ensemble, sklearn.neighbors  # noqa: F401 — импорт до таймера
    if kind == "load":                        # formats: загрузить файл и померить
        fmt, path = args[1], args[2]
        base = mem_now_mb()
        t0 = time.perf_counter()
        if fmt.startswith("pickle"):
            with open(path, "rb") as f:
                m = pickle.load(f)
        else:
            m = joblib.load(path, mmap_mode="r" if fmt == "joblib+mmap" else None)
        t = time.perf_counter() - t0
        print(json.dumps({"load_s": t, "mem_extra_mb": max(0.0, mem_peak_mb() - base),
                          "type": type(m).__name__}))
    elif kind == "mmap":                      # mmap: load + строка + весь массив
        mode, path = args[1], args[2]
        t0 = time.perf_counter()
        d = joblib.load(path, mmap_mode="r" if mode == "mmap" else None)
        t1 = time.perf_counter()
        np.array(d["embeddings"][123])
        t2 = time.perf_counter()
        float(d["embeddings"].sum())
        t3 = time.perf_counter()
        print(json.dumps({"load_ms": (t1 - t0) * 1e3, "row_ms": (t2 - t1) * 1e3,
                          "all_ms": (t3 - t2) * 1e3}))


# сжатие
def part_compression():
    import joblib
    log("\n=== 1. Сжатие: методы x уровни (случайный лес ~80 МБ) ===")
    model = model_rf_big()
    methods = ["zlib", "gzip", "bz2", "lzma"]
    try:
        import lz4  # noqa: F401
        methods.append("lz4")
    except ImportError:
        log("lz4 не установлен — пропускаю (pip install lz4)")
    rows = []
    path = TMP / "c.bin"

    def one(method, level):
        comp = 0 if method == "none" else (method, level)
        d = adaptive(lambda: joblib.dump(model, path, compress=comp))
        l = adaptive(lambda: joblib.load(path))
        row = {"method": method, "level": level, "size_mb": round(path.stat().st_size / 2**20, 2),
               "dump_s": round(d, 4), "load_s": round(l, 4)}
        rows.append(row)
        write_csv(RES / "compression.csv", rows)
        log(f"{method:>5}-{level}: {row['size_mb']:8.2f} МБ  dump {d:7.3f} c  load {l:6.3f} c")

    one("none", 0)
    for m in methods:
        for lv in range(1, 10):
            one(m, lv)
    path.unlink(missing_ok=True)

    import plot_results
    plot_results.plot_compression()


#  представление
def part_mmap(repeats=5):
    import joblib
    import numpy as np
    log("\n=== 2. mmap_mode: load + обращение к данным (массив ~305 МБ) ===")
    path = TMP / "big_array.joblib"
    joblib.dump({"embeddings": np.random.default_rng(0).random((20_000, 2_000))}, path)
    rows = []
    for i in range(repeats):
        for mode in ["full", "mmap"]:
            r = run_child("mmap", mode, str(path))
            r = {"run": i + 1, "mode": mode, **{k: round(v, 3) for k, v in r.items()}}
            rows.append(r)
            log(f"run {i+1} {mode:>4}: load {r['load_ms']:8.2f} мс + строка {r['row_ms']:6.3f} мс"
                f" | весь массив {r['all_ms']:7.1f} мс")
    write_csv(RES / "mmap.csv", rows)
    path.unlink(missing_ok=True)


# сравнение библиотек
def part_formats(repeats=3):
    import joblib
    import psutil
    log("\n=== 3. pickle vs joblib на трёх масштабах ===")
    avail = psutil.virtual_memory().available / 2**30
    free = psutil.disk_usage(str(HERE)).free / 2**30
    knn_rows = 1_300_000 if (avail >= 6 and free >= 5) else 650_000  # ~1 ГБ или ~0.5 ГБ
    log(f"Свободно RAM {avail:.1f} ГБ, диска {free:.1f} ГБ -> kNN на {knn_rows:,} строк")
    scales = [("rf_small", model_rf_small), ("rf_big", model_rf_big),
              ("knn_big", lambda: model_knn(knn_rows))]
    rows = []
    for name, make in scales:
        model = make()
        files = {"pickle-p4": TMP / f"{name}.p4", "pickle-p5": TMP / f"{name}.p5",
                 "joblib": TMP / f"{name}.joblib"}
        dumps = {}
        for fmt, p in files.items():
            if fmt.startswith("pickle"):
                proto = 4 if fmt.endswith("4") else 5
                def f(p=p, proto=proto):
                    with open(p, "wb") as fh:
                        pickle.dump(model, fh, protocol=proto)
            else:
                def f(p=p):
                    joblib.dump(model, p)
            dumps[fmt] = adaptive(f, max_n=3, slow=1.0)
        size = files["joblib"].stat().st_size / 2**20
        del model
        for fmt in ["pickle-p4", "pickle-p5", "joblib", "joblib+mmap"]:
            p = files["joblib" if fmt == "joblib+mmap" else fmt]
            loads = [run_child("load", fmt, str(p)) for _ in range(repeats)]
            row = {"scale": name, "size_mb": round(size, 1), "format": fmt,
                   "dump_s": round(dumps.get(fmt, float("nan")), 4),
                   "load_s": round(min(r["load_s"] for r in loads), 4),
                   "mem_extra_mb": round(min(r["mem_extra_mb"] for r in loads), 1)}
            rows.append(row)
            write_csv(RES / "formats.csv", rows)
            log(f"{name:>8} {size:7.1f} МБ {fmt:>11}: dump {row['dump_s']:7.3f} c"
                f"  load {row['load_s']:7.3f} c  +RAM {row['mem_extra_mb']:7.1f} МБ")
        for p in files.values():
            p.unlink(missing_ok=True)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2:])
        sys.exit()
    quiet_lz4()
    RES.mkdir(exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["compression", "mmap", "formats"]
    info = system_info()
    (RES / "system.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Система:", json.dumps(info, ensure_ascii=False))
    t0 = time.perf_counter()
    for p in parts:
        {"compression": part_compression, "mmap": part_mmap, "formats": part_formats}[p]()
    log(f"\nГотово за {(time.perf_counter() - t0) / 60:.1f} мин. Результаты: {RES}")

"""Строит графики по результатам bench_all.py (папка results/).

Можно запускать отдельно, без повторных замеров:  python plot_results.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results"
# gzip = zlib + небольшой заголовок, результаты совпадают — не рисуем отдельно
COLORS = {"zlib": "#0E7C7B", "bz2": "#2E5E8C", "lzma": "#E8892B", "lz4": "#6BAA75"}


def read(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_compression():
    rows = read("compression.csv")
    base = next(r for r in rows if r["method"] == "none")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharex=True)
    panels = [("size_mb", "Размер файла, МБ", False),
              ("dump_s", "Время сохранения, с (лог. шкала)", True),
              ("load_s", "Время загрузки, с", False)]
    for ax, (key, title, log) in zip(axes, panels):
        for m, c in COLORS.items():
            pts = [r for r in rows if r["method"] == m]
            if not pts:
                continue
            lv = [int(r["level"]) for r in pts]
            ax.plot(lv, [float(r[key]) for r in pts], marker="o", ms=4, lw=2, color=c, label=m)
        ax.axhline(float(base[key]), color="#8A97A8", ls="--", lw=1) if key != "size_mb" else None
        ax.axvline(3, color="#C9D1DC", lw=6, alpha=0.5, zorder=0)   # подсветка уровня 3
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("уровень сжатия")
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#E3E8ED", lw=0.6)
    axes[1].text(1, float(base["dump_s"]) * 1.12, "без сжатия", fontsize=9, color="#8A97A8", va="bottom")
    axes[2].set_ylim(0, None)
    axes[2].text(9, float(base["load_s"]) - 0.015, "без сжатия", fontsize=9, color="#8A97A8",
                 va="top", ha="right")
    from matplotlib.ticker import FuncFormatter
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"Случайный лес {float(base['size_mb']):.0f} МБ без сжатия; серая полоса — уровень 3",
                 x=0.01, ha="left", fontsize=11, color="#5A6675")
    fig.tight_layout()
    fig.savefig(RES / "chart_compression_levels.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_compression()
    print("Графики обновлены в", RES)

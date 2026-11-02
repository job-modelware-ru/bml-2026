"""Рисование матриц с подписанными числами — для численного примера и схем.

Все схемы свёртки, пулинга и численного примера в презентации строятся
этими функциями, поэтому их легко перерисовать с другими числами.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from common import CNN_COLOR, INK, INK_2, MUTED

CELL_EDGE = "#c9ccd3"


def draw_grid(ax, m, title=None, cmap=None, vmin=None, vmax=None, fmt="{:g}",
              highlight=None, highlight_color=CNN_COLOR, fontsize=12,
              show_values=True, zero_color="#f4f5f7", subtitle=None):
    """Нарисовать матрицу m как сетку клеток с числами.

    highlight — список прямоугольников (row, col, height, width), которые
    обводятся рамкой (например, текущее окно свёртки).
    Если cmap не задан, клетки закрашиваются светло-серым, ненулевые — белым.
    """
    m = np.asarray(m, dtype=float)
    h, w = m.shape
    ax.set_xlim(-0.05, w + 0.05)
    ax.set_ylim(h + 0.05, -0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    if cmap is not None:
        lo = m.min() if vmin is None else vmin
        hi = m.max() if vmax is None else vmax
        norm = (m - lo) / (hi - lo + 1e-12)
    for i in range(h):
        for j in range(w):
            if cmap is not None:
                face = cmap(norm[i, j])
            else:
                face = "white" if m[i, j] != 0 else zero_color
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor=CELL_EDGE, lw=1))
            if show_values:
                txt_color = INK
                if cmap is not None:
                    r, g, b, _ = face
                    txt_color = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else INK
                label = "0" if abs(m[i, j]) < 1e-12 else fmt.format(m[i, j])
                ax.text(j + 0.5, i + 0.5, label, ha="center",
                        va="center", fontsize=fontsize, color=txt_color)
    for (r, c, hh, ww) in (highlight or []):
        ax.add_patch(Rectangle((c, r), ww, hh, fill=False, edgecolor=highlight_color,
                               lw=3, zorder=5))
    if title:
        ax.set_title(title, fontsize=fontsize + 1, color=INK, pad=8)
    if subtitle:
        ax.text(w / 2, h + 0.35, subtitle, ha="center", va="top", fontsize=fontsize - 1,
                color=INK_2)
    return ax


def arrow_between(fig, ax_from, ax_to, text=None, color=MUTED):
    """Стрелка между двумя соседними осями (в координатах рисунка)."""
    b1, b2 = ax_from.get_position(), ax_to.get_position()
    y = (b1.y0 + b1.y1) / 2
    x0, x1 = b1.x1 + 0.006, b2.x0 - 0.006
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=16,
                                       color=color, lw=1.6))
    if text:
        fig.text((x0 + x1) / 2, y + 0.03, text, ha="center", va="bottom",
                 fontsize=10, color=INK_2)


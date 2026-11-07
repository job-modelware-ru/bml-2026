"""Численный пример из презентации: 14 лягушек, 2 признака, 3 итерации AdaBoost.

L — длина тела, B — яркость окраски; y = +1 — ядовитая, y = -1 — безвредная.
Слабый классификатор (решающий пень), как на слайде:
    f_{j,t,s}(x) = s,  если x_j < t;   -s,  если x_j >= t
Для каждого признака j перебираются пороги t (середины между соседними значениями)
и знак s ∈ {+1, -1}; выбирается пень с минимальной взвешенной ошибкой
    eps = sum_i w_i * [f(x_i) != y_i],   alpha = 1/2 * ln((1 - eps) / eps).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from AdaBoost import MyAdaBoost

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

names = ["Древолаз ужасный", "Древолаз клубничный", "Мантелла великолепная", "Древолаз двухцветный",
         "Квакша красноглазая", "Квакша обыкновенная", "Жаба камышовая", "Лягушка озёрная",
         "Лягушка травяная", "Лягушка-помидор", "Квакша белогубая", "Лягушка-голиаф",
         "Тростниковая жаба", "Жаба-ага яркая форма"]
L = np.array([2, 3, 2, 4, 5, 3, 2, 6, 7, 7, 8, 10, 10, 10], dtype=float)   # длина тела
B = np.array([9, 8, 7, 9, 7, 3, 2, 4, 2, 9, 8, 3, 7, 9], dtype=float)      # яркость
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1])
X = np.column_stack([L, B])
feat = ["L", "B"]
n = len(y)


def stump(x, j, t, s):
    """f_{j,t,s}(x) = s, если x_j < t, иначе -s."""
    return np.where(x[:, j] < t, s, -s)


def thresholds(col):
    v = np.unique(col)
    return (v[1:] + v[:-1]) / 2


def f4(a):
    return f"{a:.4f}"


w = np.full(n, 1 / n)
stumps, alphas, preds = [], [], []
for it in range(1, 4):
    print(f"\n{'=' * 22} Итерация {it} {'=' * 22}")
    print("веса:", " ".join(f4(v) for v in w))
    print(f"\n{'Признак':8s} {'t':>5s} {'E(s=+1)':>9s} {'E(s=-1)':>9s}")
    best = None
    for j in range(2):
        for t in thresholds(X[:, j]):
            e_plus = w[stump(X, j, t, +1) != y].sum()
            e_minus = w[stump(X, j, t, -1) != y].sum()
            print(f"{feat[j]:8s} {t:5.1f} {e_plus:9.4f} {e_minus:9.4f}")
            for s, e in ((+1, e_plus), (-1, e_minus)):
                if best is None or e < best[0] - 1e-12:
                    best = (e, j, t, s)
    eps, j, t, s = best
    h = stump(X, j, t, s)
    wrong = np.where(h != y)[0]
    alpha = 0.5 * np.log((1 - eps) / eps)
    kind = "ядовитая (+1)" if s == 1 else "безвредная (-1)"
    print(f"\nПень {it}: если {feat[j]} < {t:g} -> {kind}, иначе противоположный класс (s = {s:+d})")
    print("ошиблись:", ", ".join(f"{i + 1}" for i in wrong), "| веса ошибочных:", ", ".join(f4(w[i]) for i in wrong))
    print(f"eps_{it} = {f4(eps)} < 0.5")
    print(f"alpha_{it} = 1/2 * ln((1 - {f4(eps)}) / {f4(eps)}) = {f4(alpha)}")
    print(f"e^alpha = {f4(np.exp(alpha))},  e^-alpha = {f4(np.exp(-alpha))}")
    w_unnorm = w * np.exp(-alpha * y * h)
    Z = w_unnorm.sum()
    print(f"Z_{it} = sum w = {f4(Z)}  (= 2*sqrt(eps*(1-eps)) = {f4(2 * np.sqrt(eps * (1 - eps)))})")
    w = w_unnorm / Z
    print("новые веса:", " ".join(f4(v) for v in w))
    stumps.append((j, t, s)); alphas.append(alpha); preds.append(h)

# --- Итог: таблица как на слайде ---
F = sum(a * h for a, h in zip(alphas, preds))
y_hat = np.where(F > 0, 1, -1)
print(f"\n{'=' * 22} Итог {'=' * 22}")
print("y_hat = sign(" + " + ".join(f"{a:.4f}*f{k + 1}" for k, a in enumerate(alphas)) + ")\n")
print(f"{'№':>2s} {'Лягушка':22s} {'y':>3s} {'f1':>3s} {'f2':>3s} {'f3':>3s} {'Сумма':>8s} {'y_hat':>5s} Статус")
for i in range(n):
    print(f"{i + 1:2d} {names[i]:22s} {y[i]:+3d} {preds[0][i]:+3d} {preds[1][i]:+3d} {preds[2][i]:+3d} "
          f"{F[i]:+8.4f} {y_hat[i]:+5d} {'ok' if y_hat[i] == y[i] else 'ОШИБКА'}")
print(f"\nДоля ошибок ансамбля на обучающей выборке: {np.mean(y_hat != y):.3f}")

# --- Проверка: тот же результат даёт общий класс MyAdaBoost ---
ref = MyAdaBoost(3).fit(X, (y == 1).astype(int))
assert np.allclose(ref.alphas, alphas, atol=1e-9), (ref.alphas, alphas)
assert np.array_equal(ref.predict(X), (y_hat == 1).astype(int))
print("Проверка: MyAdaBoost(3) даёт те же alpha и те же ответы.")

# --- Рисунок: точки и разделяющие линии пней ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
gx, gy = np.meshgrid(np.linspace(0, 11, 300), np.linspace(0, 11, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])
for k, ax in enumerate(axes):
    j, t, s = stumps[k]
    ax.contourf(gx, gy, stump(grid, j, t, s).reshape(gx.shape), levels=[-2, 0, 2],
                colors=["#cfe0f5", "#f9d9c9"])
    ax.scatter(L, B, c=np.where(y == 1, "#e8743b", "#2f6fc4"), s=130, zorder=3)
    wrong = preds[k] != y
    ax.scatter(L[wrong], B[wrong], s=330, facecolors="none", edgecolors="k", linewidths=1.6, zorder=4)
    for i in range(n):
        ax.annotate(str(i + 1), (L[i], B[i]), ha="center", va="center", color="white", fontsize=7, zorder=5)
    ax.set_title(f"Пень {k + 1}: {feat[j]} < {t:g} -> {'+1' if s == 1 else '-1'},  alpha = {alphas[k]:.4f}")
    ax.set_xlabel("L — длина тела"); ax.set_ylabel("B — яркость")
    ax.set_xlim(0, 11); ax.set_ylim(0, 11)
fig.suptitle("Оранжевые — ядовитые (+1), синие — безвредные (-1); обведены ошибки пня")
plt.tight_layout(); plt.savefig(str(OUT / "simple-ex.png"), dpi=150)
plt.show()

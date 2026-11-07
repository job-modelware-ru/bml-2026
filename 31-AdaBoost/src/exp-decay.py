"""Экспоненциальное убывание ошибки обучения AdaBoost.

Теорема (Freund & Schapire): train_err <= prod_t Z_t = prod_t 2*sqrt(eps_t*(1-eps_t))
                                        <= exp(-2 * sum_t gamma_t^2),  gamma_t = 1/2 - eps_t.
Если gamma_t >= gamma > 0, то train_err <= exp(-2*gamma^2*T) — экспонента по числу итераций.
"""
from pathlib import Path
import numpy as np
from frogs import make_frogs

OUT = Path(__file__).parent / "figures"      # сюда сохраняются графики
OUT.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
from AdaBoost import MyAdaBoost

# Синтетические данные «ядовитая / безвредная лягушка» без шума в метках; классы заметно перекрываются
X, y = make_frogs(n=1000, sep=0.6, flip=0.0, seed=1)
T = 150
ada = MyAdaBoost(T).fit(X, y)
h = ada.history
ks = np.arange(1, len(h["eps"]) + 1)
gamma = 0.5 - np.array(h["eps"])
bound_prod = np.cumprod(h["Z"])                     # prod Z_t
bound_exp = np.exp(-2 * np.cumsum(gamma ** 2))      # exp(-2 sum gamma^2)
train_err = np.array(h["train_err"])
exp_loss = np.array(h["exp_loss"])

print("T    train_err   exp_loss   prod Z_t   exp(-2*sum g^2)")
for k in (1, 5, 10, 25, 50, 100, len(ks)):
    print(f"{k:3d}  {train_err[k-1]:9.4f}  {exp_loss[k-1]:9.4f}  {bound_prod[k-1]:9.4f}  {bound_exp[k-1]:9.4f}")
print("\nПроверка: exp_loss == prod Z_t ->", np.allclose(exp_loss, bound_prod))
print("Проверка: train_err <= prod Z_t  ->", bool(np.all(train_err <= bound_prod + 1e-12)))

# оценка скорости: наклон ln(exp_loss) по T
slope = np.polyfit(ks, np.log(exp_loss), 1)[0]
print(f"ln(exp_loss) ~ {slope:.4f} * T  -> ошибка падает примерно как e^({slope:.3f} T)")

plt.figure(figsize=(7, 4.2))
plt.semilogy(ks, np.maximum(train_err, 1e-4), label="ошибка обучения")
plt.semilogy(ks, exp_loss, label=r"экспоненциальная потеря $=\prod Z_t$")
plt.semilogy(ks, bound_exp, "--", label=r"оценка $\exp(-2\sum\gamma_t^2)$")
plt.xlabel("число итераций T"); plt.ylabel("значение (лог. шкала)")
plt.title("AdaBoost: ошибка убывает экспоненциально"); plt.legend(); plt.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig(str(OUT / "exp_decay.png"), dpi=150)
plt.show()

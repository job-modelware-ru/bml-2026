"""Сравнение AdaBoost (свой) с библиотечными методами на бинарной классификации."""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from AdaBoost import MyAdaBoost
from frogs import make_frogs

OUT = Path(__file__).parent / "figures"      # сюда сохраняются графики
OUT.mkdir(exist_ok=True)

# Синтетические данные «ядовитая / безвредная лягушка» (см. frogs.py): 12 признаков, 4 из них шумовые, 3% шума в метках
X, y = make_frogs(n=4000, sep=0.6, flip=0.03, seed=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

N = 100
models = {
    "Один пень":                        DecisionTreeClassifier(max_depth=1, random_state=0),
    "Логистическая регрессия":          make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest":                    RandomForestClassifier(n_estimators=N, random_state=0),
    "AdaBoost (свой, пни)":             MyAdaBoost(N),
    "AdaBoost (sklearn, пни)":          AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                           n_estimators=N, random_state=0),
    "GradBoost sklearn (пни)":          GradientBoostingClassifier(n_estimators=N, max_depth=1, random_state=0),
    "XGBoost (пни)":                    XGBClassifier(n_estimators=N, max_depth=1, learning_rate=0.3,
                                                      eval_metric="logloss", random_state=0),
    "GradBoost sklearn (глубина 3)":    GradientBoostingClassifier(n_estimators=N, max_depth=3, random_state=0),
    "XGBoost (глубина 3)":              XGBClassifier(n_estimators=N, max_depth=3, learning_rate=0.3,
                                                      eval_metric="logloss", random_state=0),
}

print(f"{'Модель':32s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'fit,с':>7s}")
for name, m in models.items():
    t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
    p = m.predict(Xte)
    s = m.decision_function(Xte) if hasattr(m, "decision_function") else m.predict_proba(Xte)[:, 1]
    print(f"{name:32s} {accuracy_score(yte, p):6.3f} {f1_score(yte, p):6.3f} "
          f"{roc_auc_score(yte, s):6.3f} {dt:7.2f}")

# --- Тестовая ошибка в зависимости от числа итераций (все на пнях) ---
ks = np.arange(1, N + 1)
my, sk = models["AdaBoost (свой, пни)"], models["AdaBoost (sklearn, пни)"]
curves = {
    "AdaBoost (свой)":  [1 - accuracy_score(yte, my.predict(Xte, k)) for k in ks],
    "AdaBoost (sklearn)": [1 - accuracy_score(yte, p) for p in sk.staged_predict(Xte)],
    "GradBoost (sklearn)": [1 - accuracy_score(yte, p) for p in models["GradBoost sklearn (пни)"].staged_predict(Xte)],
    "XGBoost": [1 - accuracy_score(yte, models["XGBoost (пни)"].predict(Xte, iteration_range=(0, k))) for k in ks],
}
plt.figure(figsize=(7, 4.2))
for name, c in curves.items():
    plt.plot(ks, c, label=name, ls="--" if "sklearn" in name and "AdaBoost" in name else "-")
plt.xlabel("число слабых моделей T"); plt.ylabel("ошибка на тесте")
plt.title("Бустинг на пнях: тестовая ошибка"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(str(OUT / "test_error_vs_T.png"), dpi=150)

# --- Устойчивость к шуму в метках ---
print("\nШум в train-метках -> test accuracy")
print(f"{'шум':>5s} {'AdaBoost':>9s} {'GradBoost':>10s} {'XGBoost':>8s}")
rng = np.random.default_rng(0)
for noise in (0.0, 0.1, 0.2, 0.3):
    yn = ytr.copy(); idx = rng.random(len(yn)) < noise; yn[idx] = 1 - yn[idx]
    r = [accuracy_score(yte, mk().fit(Xtr, yn).predict(Xte)) for mk in (
        lambda: MyAdaBoost(N),
        lambda: GradientBoostingClassifier(n_estimators=N, max_depth=1, random_state=0),
        lambda: XGBClassifier(n_estimators=N, max_depth=1, eval_metric="logloss", random_state=0))]
    print(f"{noise:5.1f} {r[0]:9.3f} {r[1]:10.3f} {r[2]:8.3f}")

# --- Совпадение собственной реализации с sklearn ---
agree = np.mean(my.predict(Xte) == sk.predict(Xte))
print(f"\nСовпадение предсказаний свой AdaBoost / sklearn AdaBoost: {agree:.3f}")
plt.show()

"""Пример 2. Загрузка сохранённой модели и предсказание без переобучения.

Запускать после 01_train_and_save.py.
"""
import json
import warnings
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer

MODELS = Path(__file__).parent / "models"

# Проверяем, что версия sklearn совпадает с той, где модель сохранили
meta = json.loads((MODELS / "model_meta.json").read_text(encoding="utf-8"))
if meta["sklearn_version"] != sklearn.__version__:
    warnings.warn(f"Модель сохранена в sklearn {meta['sklearn_version']}, "
                  f"а сейчас {sklearn.__version__} — результат может отличаться")

# Загружаем
model = joblib.load(MODELS / "model.joblib")
print("Тип:       ", type(model).__name__)
print("Шаги:      ", list(model.named_steps))

X, y = load_breast_cancer(return_X_y=True)   # 0 — злокачественная, 1 — доброкачественная
print("predict:   ", model.predict(X[:5]))   # предсказанный класс
print("истинные:  ", y[:5])                  # правильные ответы
proba = model.predict_proba(X[:5])           # вероятности классов
print("P(класс 1):", proba[:, 1].round(3))

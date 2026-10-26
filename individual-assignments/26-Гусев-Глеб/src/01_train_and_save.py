"""Пример 1. Обучение модели и сохранение её на диск с помощью joblib."""
import json
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = Path(__file__).parent / "models"
MODELS.mkdir(exist_ok=True)

# 1. Данные: 569 опухолей, 30 признаков, 2 класса (доброкачественная/злокачественная)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pipeline: масштабирование + случайный лес. Сохраняем ВСЁ вместе,
#    иначе при загрузке придётся отдельно восстанавливать scaler.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

t0 = time.perf_counter()
pipe.fit(X_train, y_train)
train_time = time.perf_counter() - t0
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"Обучение заняло {train_time:.2f} с, accuracy = {acc:.4f}")

# 3. Сохранение одной строкой
t0 = time.perf_counter()
joblib.dump(pipe, MODELS / "model.joblib")
print(f"joblib.dump: {time.perf_counter() - t0:.3f} с, "
      f"размер {(MODELS / 'model.joblib').stat().st_size / 1024:.0f} КБ")

# 4. Сохранение со сжатием (уровень 3 — разумный компромисс)
joblib.dump(pipe, MODELS / "model_compressed.joblib", compress=3)
print(f"со сжатием:  размер "
      f"{(MODELS / 'model_compressed.joblib').stat().st_size / 1024:.0f} КБ")

# 5. Метаданные рядом с моделью: версии и метрика — пригодится при загрузке
meta = {
    "sklearn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
    "n_features": int(X.shape[1]),
    "test_accuracy": round(acc, 4),
}
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Метаданные:", meta)

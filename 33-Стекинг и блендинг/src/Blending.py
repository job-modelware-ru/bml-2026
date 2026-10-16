import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=7
)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25, random_state=1, stratify=y_train_val
)

models = [
    DecisionTreeClassifier(max_depth=4, random_state=1),
    LogisticRegression(max_iter=3000),
    SVC(probability=True, random_state=1)
]

for model in models:
    model.fit(X_train, y_train)

meta_val = np.hstack([
    model.predict_proba(X_val) for model in models
])

meta_test = np.hstack([
    model.predict_proba(X_test) for model in models
])

# Метамодель обучается на прогнозах Validation
meta_model = LogisticRegression(max_iter=3000)
meta_model.fit(meta_val, y_val)

# Итоговое предсказание на тестовой выборке
y_pred = meta_model.predict(meta_test)

print("Blending accuracy:",
      accuracy_score(y_test, y_pred))

names = [
    "Decision Tree",
    "Logistic Regression",
    "SVC"
]

scores = {
    name: accuracy_score(y_test, model.predict(X_test))
    for name, model in zip(names, models)
}

scores["Blending"] = accuracy_score(y_test, y_pred)

# Выводим результаты
for name, score in scores.items():
    print(f"{name}: {score:.2%}")
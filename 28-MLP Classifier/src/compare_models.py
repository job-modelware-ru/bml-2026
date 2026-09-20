import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Генерация данных
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "SVM RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', gamma='scale')
    ),
    "KNN k=15": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15)
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "MLP (16,8)": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42
        )
    )
}

for name, model in models.items():
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='accuracy'
    )
    print(f"{name:20s} accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
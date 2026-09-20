import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. Генерация данных
# ------------------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.25, random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Размер train:", X_train_scaled.shape)
print("Размер test:", X_test_scaled.shape)

# ------------------------------------------------------------
# 2. Обучение MLPClassifier
# ------------------------------------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.01,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
    verbose=False
)

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print("Точность:", accuracy_score(y_test, y_pred))
print("Классы:", mlp.classes_)
print("Число итераций:", mlp.n_iter_)
print("Финальный loss:", mlp.loss_)
print(classification_report(y_test, y_pred))

print("W1 shape:", mlp.coefs_[0].shape)
print("b1 shape:", mlp.intercepts_[0].shape)
print("W2 shape:", mlp.coefs_[1].shape)
print("b2 shape:", mlp.intercepts_[1].shape)
print("W3 shape:", mlp.coefs_[2].shape)
print("b3 shape:", mlp.intercepts_[2].shape)

# ------------------------------------------------------------
# 3. Визуализация границы решения
# ------------------------------------------------------------
xx, yy = np.meshgrid(
    np.linspace(X_train_scaled[:, 0].min() - 0.5,
                X_train_scaled[:, 0].max() + 0.5, 300),
    np.linspace(X_train_scaled[:, 1].min() - 0.5,
                X_train_scaled[:, 1].max() + 0.5, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)

plt.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    c='blue', edgecolor='k', label='класс 0 train'
)
plt.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    c='red', edgecolor='k', label='класс 1 train'
)

plt.scatter(
    X_test_scaled[y_test == 0, 0],
    X_test_scaled[y_test == 0, 1],
    c='blue', marker='x', s=80, label='класс 0 test'
)
plt.scatter(
    X_test_scaled[y_test == 1, 0],
    X_test_scaled[y_test == 1, 1],
    c='red', marker='x', s=80, label='класс 1 test'
)

plt.title('MLPClassifier: граница решения')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.tight_layout()
plt.show()
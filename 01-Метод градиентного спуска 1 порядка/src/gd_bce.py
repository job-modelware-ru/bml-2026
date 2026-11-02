import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# 1. Генерация датасета (два класса с небольшим пересечением)
X_np, y_np = make_blobs(n_samples=1000, centers=2, n_features=2,
                        random_state=42, cluster_std=1.5)

# Конвертация в тензоры PyTorch
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)


# 2. Определение модели (Логистическая регрессия = 1 слой + Сигмоида)
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = LogisticRegressionModel()

# 3. Настройки градиентного спуска
criterion = nn.BCELoss()  # Binary Cross Entropy (строго выпуклая функция)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. Цикл обучения
epochs = 500
loss_history = []

for epoch in range(epochs):
    # Прямой проход (forward)
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    loss_history.append(loss.item())

    # Обратное распространение и шаг градиентного спуска (backward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Эпоха {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# 5. Отрисовка результатов (для слайда)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Кривая падения ошибки
ax1.plot(loss_history, color='blue', linewidth=2)
ax1.set_title('Траектория функции потерь (BCE Loss)', fontsize=14)
ax1.set_xlabel('Эпохи обучения', fontsize=12)
ax1.set_ylabel('Значение ошибки', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# График 2: Разделяющая гиперплоскость (Decision Boundary)
ax2.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='bwr', alpha=0.6, edgecolors='k')

# Получаем обученные веса из модели
w = model.linear.weight.detach().numpy()[0]
b = model.linear.bias.detach().numpy()[0]

# Строим линию: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
x1_min, x1_max = X_np[:, 0].min(), X_np[:, 0].max()
x1_line = np.array([x1_min, x1_max])
x2_line = -(w[0] * x1_line + b) / w[1]

ax2.plot(x1_line, x2_line, color='black', linewidth=3, label='Разделяющая граница')
ax2.set_title('Результат работы градиентного спуска', fontsize=14)
ax2.set_xlabel('Признак 1', fontsize=12)
ax2.set_ylabel('Признак 2', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Сохраняем график в высоком разрешении
plt.savefig('logistic_regression_bce.png', dpi=300)
plt.show()


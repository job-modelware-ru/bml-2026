import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 1. Генерируем данные с "оврагом"
torch.manual_seed(42)
X = torch.randn(1000, 2)
X[:, 0] *= 4.0  # Растягиваем первый признак
w_true = torch.tensor([[3.0], [3.0]])
y = X @ w_true + torch.randn(1000, 1) * 0.5

dataset = TensorDataset(X, y)


# 2. Функция для прогона алгоритма
def get_trajectory(batch_size, lr, epochs):
    model = nn.Linear(2, 1, bias=False)
    model.weight.data = torch.tensor([[-8.0, -8.0]])

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trajectory = [model.weight.data.clone().numpy()[0]]

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            loss = nn.MSELoss()(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            trajectory.append(model.weight.data.clone().numpy()[0])

    return np.array(trajectory)


# 3. Собираем 4 разные траектории
# Batch GD с большим шагом (эффект осцилляции/overshooting)
path_batch_large = get_trajectory(batch_size=1000, lr=0.05, epochs=50)
# Batch GD с малым шагом (гладкая траектория, но нужно больше эпох)
path_batch_small = get_trajectory(batch_size=1000, lr=0.01, epochs=150)
# Mini-batch (баланс)
path_mini = get_trajectory(batch_size=64, lr=0.01, epochs=15)
# SGD (максимум шума)
path_sgd = get_trajectory(batch_size=1, lr=0.002, epochs=2)

# 4. Вычисляем ландшафт функции потерь
w1_vals = np.linspace(-10, 5, 100)
w2_vals = np.linspace(-10, 5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = np.zeros_like(W1)
X_np, y_np = X.numpy(), y.numpy()

for i in range(100):
    for j in range(100):
        w = np.array([[W1[i, j]], [W2[i, j]]])
        Z[i, j] = np.mean((X_np @ w - y_np) ** 2)

# 5. Отрисовка
plt.figure(figsize=(12, 8))
contour = plt.contour(W1, W2, Z, levels=np.logspace(0, 3, 20), cmap='Blues', alpha=0.8)

# Рисуем линии
plt.plot(path_batch_small[:, 0], path_batch_small[:, 1], 'b-', label='Batch GD (lr=0.01) - Гладкий спуск', linewidth=3)
plt.plot(path_batch_large[:, 0], path_batch_large[:, 1], color='tab:orange', label='Batch GD (lr=0.05) - Осцилляция', linewidth=2.5)
plt.plot(path_mini[:, 0], path_mini[:, 1], 'g-x', label='Mini-batch (batch=64, lr=0.01)', linewidth=2, markersize=4)
plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'm-', label='Stochastic GD (batch=1)', linewidth=1, alpha=0.7)

plt.plot(3.0, 3.0, 'k*', markersize=15, label='Оптимум (3, 3)')

plt.title('Анализ траекторий градиентного спуска (PyTorch)', fontsize=14)
plt.xlabel('Вес $w_1$', fontsize=12)
plt.ylabel('Вес $w_2$', fontsize=12)
plt.legend(fontsize=11, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

# Сохраняем график
plt.savefig('real_gradients_combined.png', dpi=300)
plt.show()

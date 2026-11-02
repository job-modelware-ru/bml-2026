import torch
import math
import matplotlib.pyplot as plt

# 1. Подготовка данных (Синусоида)
dtype = torch.float
device = torch.device("cpu")
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 2. Инициализация весов полинома (a, b, c, d)
weights = torch.randn(4, device=device, dtype=dtype, requires_grad=True)
learning_rate = 1e-6
prev_loss = float('inf')

# 3. Цикл обучения (Градиентный спуск с Autograd)
for t in range(10000):
    # Прямой проход: y = a + bx + cx^2 + dx^3
    y_pred = weights[0] + weights[1] * x + weights[2] * (x ** 2) + weights[3] * (x ** 3)

    # Вычисление ошибки
    loss = (y_pred - y).pow(2).sum()

    # Обратный проход
    loss.backward()

    # Шаг спуска
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        weights.grad.zero_()

    if (t+1) % 500 == 499:
        print(f"Эпоха {t + 1}, Ошибка: {loss.item():.6f}")

    # Критерий остановки (Loss Tolerance)
    if abs(prev_loss - loss.item()) < 1e-8:
        print(f"Остановка на эпохе {t + 1}. Ошибка стабилизировалась: {loss.item():.6f}")
        break
    prev_loss = loss.item()

print(f"Полином: y = {weights[0].item():.3f} + {weights[1].item():.3f}x + {weights[2].item():.3f}x^2 + {weights[3].item():.3f}x^3")

# 4. Визуализация результатов
# Отвязываем тензоры от графа вычислений и переводим в numpy для отрисовки
x_plot = x.numpy()
y_true = y.numpy()
y_pred_plot = y_pred.detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, label='Истинная функция (sin)', color='blue', linewidth=2)
plt.plot(x_plot, y_pred_plot, label='Полином 3-й степени (Град. спуск)', color='red', linestyle='--', linewidth=2)

plt.title('Аппроксимация синусоиды кубическим полиномом', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Сохраняем график в высоком разрешении
plt.savefig('polynomial_approximation.png', dpi=300)
plt.show()

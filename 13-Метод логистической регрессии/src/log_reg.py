import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # сигмоида
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # обучение
    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # градиентный спуск
        for _ in range(self.epochs):

            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self._sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # вероятности для тестовой выборки
    def predict_probabilities(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    # классы для тестовой выборки
    def predict(self, X, threshold=0.5):
        return np.array([1 if p >= threshold else 0 for p in self.predict_probabilities(X)])


if __name__ == "__main__":
    # обучающая выборка
    X_train = np.array([[1, 5], [2, 6], [5, 7], [6, 8], [2, 5], [7, 6]])
    y_train = np.array([0, 0, 1, 1, 0, 1])

    # тестовая выборка
    X_test = np.array([[1.5, 5.5], [5, 6]])

    model = LogisticRegression(learning_rate=0.1, epochs=5000)
    model.fit(X_train, y_train)

    probabilities = model.predict_probabilities(X_test)
    predictions = model.predict(X_test)

    print("Результаты:")
    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        print(f"Данные: {X_test[i]}: Вероятность \"1\" = {prob:.4f} -> Итог: {pred}")


    # построение графиков
    plt.figure(figsize=(10, 6))

    X_train_class_0 = X_train[y_train == 0]
    X_train_class_1 = X_train[y_train == 1]

    plt.scatter(X_train_class_0[:, 0], X_train_class_0[:, 1], color='red', marker='o', s=100)
    plt.scatter(X_train_class_1[:, 0], X_train_class_1[:, 1], color='green', marker='o', s=100)

    for i, pred in enumerate(predictions):
        test_color = 'green' if pred == 1 else 'red'
        plt.scatter(X_test[i, 0], X_test[i, 1], color=test_color, marker='*', s=200, edgecolors='black', linewidths=1.5)

    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x1_values = np.linspace(x1_min, x1_max, 100)
    x2_values = -(model.weights[0] * x1_values + model.bias) / model.weights[1]

    plt.plot(x1_values, x2_values, color='blue', linestyle='--', linewidth=2)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(x1_min, x1_max)
    plt.ylim(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.show()
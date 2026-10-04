import random
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# Генерируем синтетические данные
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 3 * X + 5 + np.random.normal(0, 5, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

class GeneticAlgorithm:
    """Генетический алгоритм для поиска коэффициентов a и b в модели y = a*x + b"""
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_fitness_history = []
    
    def create_individual(self) -> List[float]:
        """Создаёт случайного индивидуума (пару коэффициентов a, b)"""
        return [random.uniform(-10, 10), random.uniform(-10, 10)]
    
    def create_population(self) -> List[List[float]]:
        """Создаёт начальную популяцию"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual: List[float]) -> float:
        """Вычисляет функцию приспособленности (меньше ошибка - лучше)"""
        a, b = individual
        predictions = a * X_train + b
        mse = np.mean((y_train - predictions) ** 2)
        return -mse  # Отрицательная MSE (максимизируем, т.е. минимизируем ошибку)
    
    def selection(self, population: List[List[float]]) -> List[List[float]]:
        """Турнирная селекция"""

        # Формируем новое поколение
        selected = []  # элитизм
        for _ in range(self.population_size):
            # Выбираем двух случайных особей
            tournament = random.sample(population, 2)
            # Берём ту, что лучше
            best = max(tournament, key=self.fitness)
            selected.append(best[:])
        return selected
    
    def crossover(self, parent1, parent2):
        alpha = random.random()
        child1 = [alpha * parent1[0] + (1 - alpha) * parent2[0],
              alpha * parent1[1] + (1 - alpha) * parent2[1]]
        child2 = [(1 - alpha) * parent1[0] + alpha * parent2[0],
              (1 - alpha) * parent1[1] + alpha * parent2[1]]
        return child1, child2
    
    def mutate(self, individual: List[float]) -> List[float]:
        """Мутация: случайно изменяем гены"""
        if random.random() < self.mutation_rate:
            individual[0] += random.uniform(-0.5, 0.5)
        if random.random() < self.mutation_rate:
            individual[1] += random.uniform(-0.5, 0.5)
        return individual
    
    def evolve(self) -> Tuple[List[float], float]:
        """Основной цикл эволюции"""
        population = self.create_population()
        
        for generation in range(self.generations):
            # Селекция
            selected = self.selection(population)
            # Найти лучшую особь в текущем поколении
            best_individual = max(population, key=self.fitness)
            # Скрещивание и мутация
            new_population = [best_individual[:]]
            for i in range(0, self.population_size, 2):
                child1, child2 = self.crossover(selected[i], selected[i + 1])
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            
            population = new_population[:self.population_size]
            
            # Отслеживаем лучший результат
            best = max(population, key=self.fitness)
            best_fitness = self.fitness(best)
            self.best_fitness_history.append(-best_fitness)
            
            if generation % 20 == 0:
                print(f"Поколение {generation}: лучшая ошибка = {-best_fitness:.4f}")
        
        # Возвращаем лучшего индивидуума
        best_individual = max(population, key=self.fitness)
        return best_individual, self.fitness(best_individual)

# Запуск алгоритма
ga = GeneticAlgorithm(population_size=50, generations=100, mutation_rate=0.1)
best_coefficients, best_fitness = ga.evolve()
# ГА
a, b = best_coefficients
y_pred_ga = a * X_test + b

print(f"\n✓ Лучшие найденные коэффициенты:")
print(f"  a (наклон) = {best_coefficients[0]:.4f}")
print(f"  b (смещение) = {best_coefficients[1]:.4f}")
print(f"  Исходные: a = 3.0, b = 5.0")
test_mse = np.mean((y_test - (best_coefficients[0] * X_test + best_coefficients[1])) ** 2)
print(f"Ошибка MSE на тренировочных данных: {-best_fitness:.4f}")
print(f"Ошибка MSE на тестовых данных:  {test_mse:.4f}")

mae = mean_absolute_error(y_test, y_pred_ga)
r2 = r2_score(y_test, y_pred_ga)

print(f"MAE: {mae:.4f}")
print(f"R²:  {r2:.4f}")

# sklearn
lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train)
y_pred_lr = lr.predict(X_test.reshape(-1, 1))



# Сравнение
print("=" * 60)
print("СРАВНЕНИЕ С SKLEARN")
print("=" * 60)
print(f"{'Метод':<15} {'a':>8} {'b':>8} {'Test MSE':>10} {'R²':>8}")
print(f"{'sklearn':<15} {lr.coef_[0]:>8.4f} {lr.intercept_:>8.4f} "
      f"{np.mean((y_test - y_pred_lr)**2):>10.4f} "
      f"{r2_score(y_test, y_pred_lr):>8.4f}")
print(f"{'ГА':<15} {a:>8.4f} {b:>8.4f} "
      f"{np.mean((y_test - y_pred_ga)**2):>10.4f} "
      f"{r2_score(y_test, y_pred_ga):>8.4f}")



plt.figure(figsize=(12, 5))

# График 1: данные + прямая
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Train')
plt.scatter(X_test, y_test, alpha=0.5, label='Test', marker='^')
plt.plot(X, a * X + b, 'r-', linewidth=2, 
         label=f'ГА: y = {a:.2f}x + {b:.2f}')
plt.plot(X, lr.coef_[0] * X + lr.intercept_, 'g--', linewidth=2,
         label=f'sklearn: y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('ГА vs sklearn')
plt.legend()
plt.grid(alpha=0.3)

# График 2: сходимость
plt.subplot(1, 2, 2)
plt.plot(ga.best_fitness_history, 'b-', linewidth=2)
plt.xlabel('Поколение')
plt.ylabel('MSE')
plt.title('Сходимость ГА')
plt.grid(alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('ga_regression.png', dpi=200)
plt.show()


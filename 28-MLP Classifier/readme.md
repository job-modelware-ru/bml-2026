# MLP Classifier

Команда: 28
Сагун Александр, Житков Никита
5030102/30201

## Содержание каталога

| Файл | Что внутри |
| --- | --- |
| `src/` | Исходный код |
| `MLP Classifier.pdf` | Презентация |
| `readme.md` | Описание проекта |

### Каталог `src/`

| Файл | Что внутри |
| --- | --- |
| `demo_mlp.py` | Генерирует синтетический датасет `make_moons` (500 точек, шум 0.25), делит его на train/test 70/30 со стратификацией, стандартизирует признаки через `StandardScaler`, обучает `MLPClassifier` с архитектурой `(16, 8)`, активацией ReLU и оптимизатором Adam, выводит точность, `classification_report`, число итераций, финальный loss и формы весов, затем строит график границы решения |
| `numerical_mlp.py` | Численный пошаговый пример обучения MLP только на NumPy: сеть 2 → 2 → 1, активация sigmoid, считается forward pass (`z1`, `a1`, `z2`, `a2`, loss) и backward pass (градиенты `dz2`, `dW2`, `db2`, `dz1`, `dW1`, `db1`), затем один шаг градиентного спуска с `lr = 0.5` |
| `compare_models.py` | Сравнивает MLP с альтернативными моделями на `make_moons` через 5-фолдовую кросс-валидацию: LogisticRegression, SVM с RBF-ядром, KNN (k=15), RandomForest, GradientBoosting и MLP (16, 8); выводит accuracy ± std для каждой модели |
| `requirements.txt` | Зависимости |

## Запуск кода

Все скрипты можно запускать по отдельности:

```bash
pip install -r requirements.txt

python src/demo_mlp.py          # обучение MLP на make_moons + визуализация
python src/numerical_mlp.py     # численный пример forward/backward вручную
python src/compare_models.py    # сравнение MLP с другими моделями
```
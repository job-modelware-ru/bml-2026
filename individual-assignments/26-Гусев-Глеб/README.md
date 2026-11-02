# Примеры: joblib

## Самый простой способ (Windows)

Дважды кликнуть `run_benchmarks.bat`. Он сам:
1. создаст виртуальное окружение `.venv` (отдельную папку с библиотеками только для этого проекта);
2. установит зависимости из `requirements.txt`;
3. запустит примеры 01–06 (вывод — в `output/`);
4. запустит замеры `bench_all.py` (результаты — в `results/`), это около 15–25 минут.

Требования: Python 3.9+ в PATH, ~5 ГБ свободного места и ~6 ГБ свободной RAM
(если меньше, тест kNN автоматически уменьшится до ~0,5 ГБ).
Во время замеров лучше не нагружать компьютер — иначе время «поплывёт».

## Вручную

Установка: `pip install -r requirements.txt` (Python 3.9+).

| Файл | Что показывает |
|---|---|
| 01_train_and_save.py | Обучение Pipeline (StandardScaler + RandomForest) и `joblib.dump`, в том числе с `compress=3`; сохранение метаданных (версии, accuracy) в `models/model_meta.json` |
| 02_load_and_predict.py | `joblib.load`, проверка версии scikit-learn, предсказания без повторного обучения. Запускать после 01 |
| 03_compression_benchmark.py | pickle vs joblib и разные методы/уровни сжатия: размер, время dump/load; строит графики в `output/` |
| 04_mmap_large_arrays.py | `mmap_mode="r"` на массиве 305 МБ (файл создаётся и удаляется; нужно ~1 ГБ свободной памяти) |
| 05_security_demo.py | Почему нельзя загружать чужие .joblib: код выполняется прямо при `load` (безобидный print) |
| 06_pickle_vs_joblib_arrays.py | pickle vs joblib на KNN, хранящем 116 МБ данных |
| run_all.py | Запускает примеры 01–06 по порядку и пишет вывод в `output/*.txt` |
| bench_all.py | Полные замеры для слайдов: все уровни сжатия, mmap, pickle vs joblib на 1 МБ / 80 МБ / 1 ГБ. Можно запустить часть: `python bench_all.py mmap` |
| plot_results.py | Перерисовывает графики по готовым `results/*.csv` без повторных замеров |
| run_benchmarks.bat | Всё сразу для Windows (окружение + примеры + замеры) |

### Что лежит в results/ после замеров

| Файл | Содержимое |
|---|---|
| system.json | процессор, RAM, версии Python и библиотек |
| compression.csv, chart_compression_levels.png | метод, уровень, размер, время dump/load |
| mmap.csv | 5 прогонов: load, обращение к строке, обход всего массива |
| formats.csv | pickle p4/p5, joblib, joblib+mmap: dump, load, доп. память |
| log.txt, examples_log.txt | полный вывод |

Папка `models/` создаётся при запуске (в репозиторий её класть не нужно).
Время в `output/` зависит от компьютера — на слайдах приведены замеры одного из прогонов
(Python 3.13, scikit-learn 1.9.1, joblib 1.6.0), поэтому числа могут немного отличаться.

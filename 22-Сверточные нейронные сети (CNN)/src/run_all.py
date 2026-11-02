"""Последовательный запуск всех скриптов и пересборка figures/ и results/.

    python run_all.py            все скрипты по порядку
    python run_all.py 03 04      только указанные

Обученные модели сохраняются в results/models/ и при повторном запуске
берутся с диска. Первый полный прогон на процессоре занимает около
1,5 часа (почти всё время — обучение сетей в 06 и 07), на видеокарте —
несколько минут. Быстрая проверка без долгих экспериментов:
    python run_all.py 01 02 03 04 05 08
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ("01_why_cnn.py", "Постановка задачи: картинка как данные"),
    ("02_convolution_basics.py", "Свёртка, пулинг, рецептивное поле — с нуля на NumPy"),
    ("03_numerical_example.py", "Численный пример: прямой и обратный проход вручную"),
    ("04_train_cnn.py", "Обучение SimpleCNN на Fashion-MNIST"),
    ("05_analyze_cnn.py", "Анализ обученной сети"),
    ("06_compare_models.py", "Сравнение с альтернативными моделями"),
    ("07_inductive_bias.py", "Сдвиг, перемешанные пиксели, мало данных"),
    ("08_architectures.py", "Известные архитектуры на ImageNet"),
]


def main():
    selected = sys.argv[1:]
    started = time.perf_counter()
    failed = []
    for name, title in SCRIPTS:
        if selected and not any(name.startswith(key) for key in selected):
            continue
        print("\n" + "#" * 74 + f"\n# {name} — {title}\n" + "#" * 74)
        t0 = time.perf_counter()
        code = subprocess.run([sys.executable, name], cwd=HERE).returncode
        if code:
            failed.append(name)
            print(f"{name}: ошибка (код {code})")
        else:
            print(f"{name}: выполнено за {time.perf_counter() - t0:.0f} с")

    figures = sorted((HERE / "figures").glob("*.png"))
    print("\n" + "=" * 74 + f"\nРисунков в figures/: {len(figures)}")
    for path in figures:
        print(f"  {path.name}")
    print(f"Общее время: {time.perf_counter() - started:.0f} с")
    if failed:
        print("С ошибкой завершились: " + ", ".join(failed))
        sys.exit(1)
    print("Все скрипты выполнены успешно.")


if __name__ == "__main__":
    main()

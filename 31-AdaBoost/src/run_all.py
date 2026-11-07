"""Запускает все примеры по очереди: пошаговый пример, экспоненциальное убывание, сравнение методов."""
import runpy
from pathlib import Path

here = Path(__file__).parent
for name in ("simple-ex.py", "exp-decay.py", "compare-methods.py"):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    runpy.run_path(str(here / name), run_name="__main__")

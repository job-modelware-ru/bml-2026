"""Запускает все примеры по очереди и сохраняет их вывод в output/."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

for script in sorted(HERE.glob("0*.py")):
    print(f"===== {script.name} =====")
    res = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, encoding="utf-8", cwd=HERE,
                         env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
    text = res.stdout + res.stderr
    print(text)
    (OUT / f"{script.stem}.txt").write_text(text, encoding="utf-8")

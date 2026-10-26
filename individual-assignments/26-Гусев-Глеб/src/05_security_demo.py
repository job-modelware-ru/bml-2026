"""Пример 5. Почему нельзя загружать .joblib/.pkl из недоверенных источников.

joblib основан на pickle. При загрузке pickle может вызвать ЛЮБУЮ функцию,
которую указал автор файла (через метод __reduce__). Здесь вместо вредного
кода — безобидный print, но на его месте мог быть os.system(...).
"""
from pathlib import Path

import joblib

PATH = Path(__file__).parent / "models" / "suspicious_model.joblib"
PATH.parent.mkdir(exist_ok=True)


MSG = "!!! Этот код выполнился при joblib.load() !!!"


class FakeModel:
    def __reduce__(self):            # вызывается pickle при сохранении
        return (print, (MSG,))       # «при загрузке вызови print(MSG)»


joblib.dump(FakeModel(), PATH)
print("Файл создан. Загружаем его...")
obj = joblib.load(PATH)          # здесь и сработает «закладка»
print("Что вернул load:", obj)   # None — модели внутри даже не было

"""Известные архитектуры CNN и Vision Transformer на ImageNet.

Числа берутся из метаданных предобученных весов torchvision (скачивать
сами веса не нужно): число параметров, вычислительная стоимость (GFLOPS
на одну картинку 224×224) и точность top-1 на ImageNet-1K (1000 классов,
50 000 проверочных картинок). Это результаты рецептов обучения
torchvision, они могут немного отличаться от чисел в исходных статьях.
"""

import matplotlib.pyplot as plt
from torchvision.models import get_weight

from common import CNN_COLOR, INK, INK_2, MLP_COLOR, save, save_json, section

# (название, год публикации, идентификатор весов, ключевая идея)
ARCHS = [
    ("AlexNet", 2012, "AlexNet_Weights.IMAGENET1K_V1", "ReLU, Dropout, обучение на GPU"),
    ("VGG-16", 2014, "VGG16_Weights.IMAGENET1K_V1", "только свёртки 3×3, глубина 16"),
    ("GoogLeNet", 2014, "GoogLeNet_Weights.IMAGENET1K_V1", "Inception: параллельные ядра 1×1, 3×3, 5×5"),
    ("ResNet-50", 2015, "ResNet50_Weights.IMAGENET1K_V1", "остаточные связи y = F(x) + x"),
    ("ResNet-152", 2015, "ResNet152_Weights.IMAGENET1K_V1", "152 слоя благодаря остаточным связям"),
    ("DenseNet-121", 2016, "DenseNet121_Weights.IMAGENET1K_V1", "каждый слой получает выходы всех предыдущих"),
    ("MobileNetV2", 2018, "MobileNet_V2_Weights.IMAGENET1K_V1", "depthwise-свёртки для телефонов"),
    ("EfficientNet-B0", 2019, "EfficientNet_B0_Weights.IMAGENET1K_V1", "согласованное масштабирование глубины/ширины/разрешения"),
    ("ViT-B/16", 2020, "ViT_B_16_Weights.IMAGENET1K_V1", "трансформер на патчах 16×16, без свёрток"),
    ("ConvNeXt-T", 2022, "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "CNN, модернизированная по образцу ViT"),
]


def main():
    section("Архитектуры на ImageNet-1K (метаданные torchvision)")
    rows = []
    print(f"  {'модель':<16}{'год':>5}{'параметров, млн':>17}{'GFLOPS':>8}{'top-1, %':>10}")
    for name, year, weights, idea in ARCHS:
        meta = get_weight(weights).meta
        row = {"name": name, "year": year, "params_m": meta["num_params"] / 1e6,
               "gflops": meta["_ops"], "top1": meta["_metrics"]["ImageNet-1K"]["acc@1"],
               "top5": meta["_metrics"]["ImageNet-1K"]["acc@5"], "idea": idea}
        rows.append(row)
        print(f"  {name:<16}{year:>5}{row['params_m']:>17.1f}{row['gflops']:>8.2f}{row['top1']:>10.2f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for r in rows:
        is_vit = r["name"].startswith("ViT")
        ax.scatter(r["params_m"], r["top1"], s=40 + 18 * r["gflops"],
                   color=MLP_COLOR if is_vit else CNN_COLOR, alpha=0.8,
                   edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(f"{r['name']} ({r['year']})", (r["params_m"], r["top1"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=9.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("число параметров, млн (лог. шкала)")
    ax.set_ylabel("точность top-1 на ImageNet, %")
    ax.set_title("Размер кружка — вычислительная стоимость (GFLOPS); "
                 "оранжевые — CNN, синий — трансформер", fontsize=10.5, color=INK)
    ax.set_xlim(2, 400)
    fig.tight_layout()
    save(fig, "architectures")
    save_json(rows, "architectures")


if __name__ == "__main__":
    main()

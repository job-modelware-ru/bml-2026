from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    profiles = {
        "city": {
            "engine": (1.6, 0.45), "power": (125, 50), "weight": (1350, 180),
            "length": (4.25, 0.3), "doors": (4.5, 0.5), "seats": (5, 0.6),
            "fuel": (7, 1.4), "age": (6, 4), "mileage": (90000, 50000), "price": (22, 10),
        },
        "family": {
            "engine": (2.0, 0.5), "power": (170, 60), "weight": (1600, 220),
            "length": (4.6, 0.3), "doors": (4.5, 0.5), "seats": (5.5, 1),
            "fuel": (8, 1.5), "age": (5, 3.5), "mileage": (75000, 45000), "price": (30, 12),
        },
        "sport": {
            "engine": (2.6, 0.8), "power": (240, 100), "weight": (1500, 250),
            "length": (4.4, 0.35), "doors": (3.2, 0.8), "seats": (3.8, 1),
            "fuel": (10, 2.5), "age": (4, 3.5), "mileage": (60000, 40000), "price": (42, 20),
        },
    }

    rows = []
    for segment, profile in profiles.items():
        for _ in range(500):
            rows.append(
                {
                    "engine_volume_l": rng.normal(*profile["engine"]),
                    "horsepower": rng.normal(*profile["power"]),
                    "weight_kg": rng.normal(*profile["weight"]),
                    "length_m": rng.normal(*profile["length"]),
                    "doors": np.rint(rng.normal(*profile["doors"])),
                    "seats": np.rint(rng.normal(*profile["seats"])),
                    "fuel_consumption_l_per_100km": rng.normal(*profile["fuel"]),
                    "age_years": rng.normal(*profile["age"]),
                    "mileage_km": rng.normal(*profile["mileage"]),
                    "price_thousand": rng.normal(*profile["price"]),
                    "segment": segment,
                }
            )

    data = pd.DataFrame(rows)
    data["engine_volume_l"] = data["engine_volume_l"].clip(0.7, 6).round(2)
    data["horsepower"] = data["horsepower"].clip(45, 600).round(1)
    data["weight_kg"] = data["weight_kg"].clip(750, 3000).round(1)
    data["length_m"] = data["length_m"].clip(3.2, 5.8).round(2)
    data["doors"] = data["doors"].clip(2, 5).astype(int)
    data["seats"] = data["seats"].clip(2, 8).astype(int)
    data["fuel_consumption_l_per_100km"] = data["fuel_consumption_l_per_100km"].clip(3.5, 20).round(2)
    data["age_years"] = data["age_years"].clip(0, 20).round(1)
    data["mileage_km"] = data["mileage_km"].clip(0, 300000).round().astype(int)
    data["price_thousand"] = data["price_thousand"].clip(5, 150).round(1)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    output_path = Path(__file__).with_name("car_dataset.csv")
    data.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()

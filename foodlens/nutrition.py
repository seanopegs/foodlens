"""Nutrition estimates for a typical serving of each Food-101 dish.

Values come from USDA FNDDS 2021-2023 and are built into nutrition.json by
nutrition/build_nutrition.py. The app scales them by portion size and adds the
extras the user picks (extra oil, cheese, rice, ...).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

NUTRIENTS = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg")
PORTIONS = {"Small": 0.7, "Regular": 1.0, "Large": 1.4}
DAILY_KCAL = 2000
TABLE_PATH = Path(__file__).with_name("nutrition.json")


@lru_cache(maxsize=1)
def load_table(path: str | None = None) -> dict:
    with open(path or TABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def estimate(dish: str, portion: float = 1.0, extras: dict[str, int] | None = None,
             extra_kcal: float = 0.0, table: dict | None = None) -> dict[str, float]:
    """Nutrients for ``portion`` x the typical serving of ``dish`` plus extras.

    ``extras`` maps add-on keys (see nutrition.json) to how many units were added.
    ``extra_kcal`` covers anything else the user adds; it only changes calories.
    """
    table = table or load_table()
    base = table["dishes"][dish]["per_serving"]
    total = {k: base[k] * portion for k in NUTRIENTS}
    for key, count in (extras or {}).items():
        unit = table["addons"][key]["per_unit"]
        for k in NUTRIENTS:
            total[k] += unit[k] * count
    total["kcal"] += extra_kcal
    return total


def energy_split(total: dict[str, float]) -> dict[str, float]:
    """Share of calories from protein, carbs and fat (4 / 4 / 9 kcal per gram)."""
    kcal = {"protein": 4 * total["protein_g"], "carbs": 4 * total["carbs_g"], "fat": 9 * total["fat_g"]}
    s = sum(kcal.values()) or 1.0
    return {k: v / s for k, v in kcal.items()}

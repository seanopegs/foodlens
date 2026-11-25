"""Build foodlens/nutrition.json from the USDA FNDDS tables.

Usage:
    python nutrition/build_nutrition.py <path to FoodData_Central_survey_food_csv folder>

Download the "Survey (FNDDS)" CSV release from https://fdc.nal.usda.gov/download-datasets
(we used FoodData_Central_survey_food_csv_2024-10-31, i.e. FNDDS 2021-2023) and unzip it.
servings.csv maps every Food-101 class to one or more FNDDS foods with a gram weight for a
typical serving; addons.csv lists the extras a user can add in the app.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "foodlens" / "nutrition.json"

# FDC nutrient id and the older nutrient number; the survey tables use the number.
NUTRIENTS = {
    "kcal": ("1008", "208"),
    "protein_g": ("1003", "203"),
    "carbs_g": ("1005", "205"),
    "fat_g": ("1004", "204"),
    "fiber_g": ("1079", "291"),
    "sugar_g": ("2000", "269"),
    "sodium_mg": ("1093", "307"),
}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(fndds_dir: str) -> None:
    src = Path(fndds_dir)
    names = {r["fdc_id"]: r["description"] for r in read_csv(src / "food.csv")}
    code_to_key = {code: key for key, codes in NUTRIENTS.items() for code in codes}

    servings = read_csv(HERE / "servings.csv")
    addons = read_csv(HERE / "addons.csv")
    needed = {r["fdc_id"] for r in servings + addons}

    per_100g: dict[str, dict[str, float]] = defaultdict(dict)
    with open(src / "food_nutrient.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = code_to_key.get(r["nutrient_id"])
            if key and r["fdc_id"] in needed:
                per_100g[r["fdc_id"]][key] = float(r["amount"])

    missing = needed - set(per_100g)
    if missing:
        raise SystemExit(f"FNDDS ids not found: {sorted(missing)}")

    def amount(fdc_id: str, grams: float) -> dict[str, float]:
        values = per_100g[fdc_id]
        return {k: values.get(k, 0.0) * grams / 100 for k in NUTRIENTS}

    dishes: dict[str, dict] = {}
    for r in servings:
        dish = dishes.setdefault(r["dish"], {
            "serving": r["serving"], "grams": 0.0, "match": r["match"], "based_on": [],
            "per_serving": dict.fromkeys(NUTRIENTS, 0.0),
        })
        grams = float(r["grams"])
        dish["grams"] += grams
        dish["based_on"].append(f'{names[r["fdc_id"]]} ({grams:g} g, FDC {r["fdc_id"]})')
        for k, v in amount(r["fdc_id"], grams).items():
            dish["per_serving"][k] += v

    extras = {}
    for r in addons:
        grams = float(r["grams"])
        extras[r["key"]] = {
            "label": r["label"], "unit": r["unit"], "grams": grams,
            "based_on": f'{names[r["fdc_id"]]} (FDC {r["fdc_id"]})',
            "per_unit": amount(r["fdc_id"], grams),
        }

    def rounded(d: dict) -> dict:
        return {k: round(v, 1) for k, v in d.items()}

    for d in dishes.values():
        d["grams"] = round(d["grams"])
        d["per_serving"] = rounded(d["per_serving"])
    for e in extras.values():
        e["per_unit"] = rounded(e["per_unit"])

    out = {
        "source": "USDA Food and Nutrient Database for Dietary Studies (FNDDS) 2021-2023, "
                  "FoodData Central survey foods release 2024-10-31",
        "dishes": dict(sorted(dishes.items())),
        "addons": extras,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(dishes)} dishes, {len(extras)} add-ons)")


if __name__ == "__main__":
    main(sys.argv[1])

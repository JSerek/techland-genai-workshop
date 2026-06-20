#!/usr/bin/env python3
"""Buduje finalny golden dataset (20 negatywnych recenzji, multi-label, 15 kategorii).

Mapa `review_id -> labels` jest zaszyta dokładnie wg `analysis/golden_proposed.md`
(sekcje 1-20, pola "LABELS"). Pełne `review_text` i `votes_up` wczytywane są z
`analysis/golden_candidates.csv` po `review_id` (nie używamy skróconych tekstów z .md).

Wyjście: nadpisuje `data/evaluation/golden_dataset.json` i tworzy `.csv`.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "analysis" / "golden_candidates.csv"
OUT_JSON = ROOT / "data" / "evaluation" / "golden_dataset.json"
OUT_CSV = ROOT / "data" / "evaluation" / "golden_dataset.csv"

# 15 kategorii (kody) — źródło: analysis/aspects_with_definitions.md
CATEGORIES = {
    "combat", "parkour", "enemies", "night_horror", "progression", "world",
    "story", "bugs", "performance", "graphics", "audio", "content", "price",
    "coop", "gore",
}

# review_id -> labels (kolejność wg golden_proposed.md, sekcje 1-20)
GOLDEN_LABELS = {
    "205490944": ["content", "price"],
    "204662234": ["combat", "enemies", "price"],
    "204658386": ["progression", "content", "price"],
    "204958301": ["combat", "enemies"],
    "217659097": ["world", "content"],
    "204834661": ["content", "combat", "enemies"],
    "204957991": ["price", "content", "world", "story", "combat"],
    "204639308": ["combat", "parkour", "progression", "price"],
    "204632770": ["graphics", "combat", "gore", "progression", "bugs", "price"],
    "204870501": ["content", "price"],
    "206740572": ["story", "content", "price"],
    "204878716": ["bugs", "performance"],
    "204642036": ["story", "performance"],
    "205076487": ["story", "world", "combat", "parkour"],
    "205100534": ["world"],
    "204641037": ["graphics", "gore", "progression", "story"],
    "204926978": ["progression", "combat", "parkour", "story", "content"],
    "205334075": ["coop", "bugs"],
    "207836299": ["night_horror", "world", "enemies", "combat", "story", "price"],
    "215810589": ["story", "audio", "parkour", "combat", "enemies", "progression", "world"],
}

# Oczekiwany rozkład pokrycia (z nagłówka golden_proposed.md) — kontrola sanity.
EXPECTED_COVERAGE = {
    "combat": 10, "price": 9, "content": 8, "story": 8, "progression": 6,
    "world": 6, "enemies": 5, "parkour": 4, "bugs": 3, "performance": 2,
    "graphics": 2, "gore": 2, "night_horror": 1, "audio": 1, "coop": 1,
}


def load_source():
    """Wczytaj review_text + votes_up z golden_candidates.csv po review_id."""
    by_id = {}
    with open(CANDIDATES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["review_id"]] = row
    return by_id


def build():
    source = load_source()

    records = []
    for i, (review_id, labels) in enumerate(GOLDEN_LABELS.items(), start=1):
        assert review_id in source, f"review_id {review_id} nie znaleziony w {CANDIDATES.name}"
        src = source[review_id]
        records.append({
            "id": i,
            "review_id": review_id,
            "votes_up": int(src["votes_up"]),
            "review_text": src["review_text"],
            "labels": labels,
        })

    # --- Walidacja (twarde asercje) ---
    assert len(records) == 20, f"oczekiwano 20 rekordów, jest {len(records)}"
    ids = [r["review_id"] for r in records]
    assert len(set(ids)) == 20, "duplikat review_id"
    for r in records:
        for label in r["labels"]:
            assert label in CATEGORIES, f"etykieta '{label}' spoza 15 kategorii (rec. {r['review_id']})"

    # Kontrola rozkładu pokrycia
    coverage = {}
    for r in records:
        for label in r["labels"]:
            coverage[label] = coverage.get(label, 0) + 1
    assert coverage == EXPECTED_COVERAGE, (
        f"rozkład pokrycia niezgodny z golden_proposed.md:\n"
        f"  policzony: {dict(sorted(coverage.items()))}\n"
        f"  oczekiwany: {dict(sorted(EXPECTED_COVERAGE.items()))}"
    )

    # --- Zapis JSON ---
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # --- Zapis CSV (labels łączone ';') ---
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "review_id", "votes_up", "review_text", "labels"])
        for r in records:
            writer.writerow([
                r["id"], r["review_id"], r["votes_up"],
                r["review_text"], ";".join(r["labels"]),
            ])

    print(f"OK: zapisano {len(records)} rekordów")
    print(f"  -> {OUT_JSON.relative_to(ROOT)}")
    print(f"  -> {OUT_CSV.relative_to(ROOT)}")
    print(f"Pokrycie aspektów: {dict(sorted(coverage.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    build()

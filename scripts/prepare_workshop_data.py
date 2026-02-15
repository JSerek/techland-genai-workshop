"""
Przygotowanie danych do warsztatów.

Generuje:
1. data/processed/workshop_sample.csv  - próbka recenzji do ćwiczeń
2. data/evaluation/golden_dataset.json - mock dataset (do uzupełnienia ręcznie)

Użycie:
    python scripts/prepare_workshop_data.py
    python scripts/prepare_workshop_data.py --sample-size 30 --min-length 80
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def prepare_sample(
    input_csv: str,
    output_path: str,
    sample_size: int = 50,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Tworzy próbkę roboczą z danych scrapera."""
    df = pd.read_csv(input_csv)
    print(f"Załadowano {len(df):,} recenzji z {input_csv}")

    df["review_length"] = df["review_text"].str.len()
    df_filtered = df[
        (df["review_length"] >= min_length) &
        (df["review_length"] <= max_length) &
        (df["review_text"].notna())
    ].copy()
    print(f"Po filtrowaniu: {len(df_filtered):,} recenzji")

    n = min(sample_size, len(df_filtered))
    df_sample = df_filtered.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"Próbka: {n} recenzji")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    print(f"✅ Zapisano: {output_path}")

    return df_sample


def prepare_golden_dataset(
    df_sample: pd.DataFrame,
    output_path: str,
    n_examples: int = 20,
    categories: list[str] | None = None,
    seed: int = 42,
) -> None:
    """
    Tworzy szkielet golden dataset do ręcznego etykietowania.

    Jeśli categories jest None, tworzy plik z pustymi etykietami do wypełnienia.
    Jeśli categories jest podane, losowo przypisuje mock etykiety (tylko do testów technicznych!).
    """
    if categories is None:
        categories = []

    random.seed(seed)
    n = min(n_examples, len(df_sample))
    sample = df_sample.sample(n=n, random_state=seed)

    golden_data = []
    for _, row in sample.iterrows():
        if categories:
            # Mock: losowo przypisz 1-2 kategorie (TYLKO do testów technicznych)
            n_cats = random.randint(1, min(2, len(categories)))
            labels = random.sample(categories, n_cats)
        else:
            labels = []  # Do ręcznego wypełnienia

        golden_data.append({
            "review_id": str(row.get("review_id", "")),
            "review_text": row["review_text"],
            "labels": labels,
            "sentiment": row.get("sentiment", ""),
            # Pole do ręcznego uzupełnienia:
            "_notes": ""  # opcjonalny komentarz dla etykietującego
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, ensure_ascii=False, indent=2)

    if categories:
        print(f"✅ Mock golden dataset ({n} przykładów, LOSOWE etykiety): {output_path}")
        print("   ⚠️  UWAGA: Etykiety są losowe - tylko do testów technicznych!")
    else:
        print(f"✅ Szkielet golden dataset ({n} przykładów, puste etykiety): {output_path}")
        print("   👉 Uzupełnij pole 'labels' ręcznie przed właściwą ewaluacją.")


def main():
    parser = argparse.ArgumentParser(description="Przygotowanie danych warsztatowych")
    parser.add_argument("--input", default="data/raw/dying_light_beast_negative_full.csv")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--golden-size", type=int, default=20,
                        help="Liczba recenzji w golden dataset")
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--mock-labels", action="store_true",
                        help="Przypisz losowe mock etykiety (tylko do testów technicznych)")
    parser.add_argument("--categories", nargs="+",
                        default=None,
                        help="Kategorie do mock etykiet, np: bug performance story")
    args = parser.parse_args()

    print("=" * 60)
    print("PRZYGOTOWANIE DANYCH WARSZTATOWYCH")
    print("=" * 60)

    # 1. Próbka robocza
    df_sample = prepare_sample(
        input_csv=args.input,
        output_path="data/processed/workshop_sample.csv",
        sample_size=args.sample_size,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    # 2. Golden dataset
    categories = args.categories if args.mock_labels else None
    prepare_golden_dataset(
        df_sample=df_sample,
        output_path="data/evaluation/golden_dataset.json",
        n_examples=args.golden_size,
        categories=categories,
    )

    print("\n" + "=" * 60)
    print("GOTOWE. Następne kroki:")
    if not args.mock_labels:
        print("  1. Otwórz data/evaluation/golden_dataset.json")
        print("  2. Uzupełnij pole 'labels' dla każdej recenzji")
        print("  3. Usuń pole '_notes' lub użyj do komentarzy")
    print("  4. Uruchom scripts/test_pipeline.py (gdy masz API key)")
    print("=" * 60)


if __name__ == "__main__":
    main()

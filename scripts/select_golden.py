"""Selekcja puli kandydatów do golden datasetu (warsztat Techland).

Read-only względem danych źródłowych. Filtruje recenzje negatywne (english),
zawęża do pasa długości i zwraca pulę posortowaną po votes_up.

Wynik: analysis/golden_candidates.csv
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/dying_light_beast_negative_full.csv"
OUT = ROOT / "analysis/golden_candidates.csv"

LEN_MIN, LEN_MAX = 200, 700
POOL_SIZE = 150


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    en = [r for r in rows if r["language"] == "english"]
    for r in en:
        r["len"] = len(r["review_text"])
        r["votes_up_int"] = int(r["votes_up"] or 0)

    band = [r for r in en if LEN_MIN <= r["len"] <= LEN_MAX]
    band.sort(key=lambda r: -r["votes_up_int"])
    pool = band[:POOL_SIZE]

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "votes_up", "len", "review_text"])
        for r in pool:
            w.writerow([r["review_id"], r["votes_up_int"], r["len"],
                        " ".join(r["review_text"].split())])

    print(f"english negatives: {len(en)}")
    print(f"w pasie {LEN_MIN}-{LEN_MAX}: {len(band)}")
    print(f"zapisano pulę: {len(pool)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

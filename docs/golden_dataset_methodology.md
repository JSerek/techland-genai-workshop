# Metodologia budowy Golden Dataset

> Dokument opisuje **cały proces analityczny** prowadzący do `data/evaluation/golden_dataset.{json,csv}`
> — od surowych recenzji Steam po 20 ręcznie zaetykietowanych komentarzy multi-label.
> Decyzje formalne: patrz [D-006] w `decisions_backlog.md`. Taksonomia: `analysis/aspects_with_definitions.md`.
>
> **Data:** 2026-06-20 · **Autor procesu:** analiza wspólna (prowadzący + asystent)

---

## 1. Cel i ograniczenia

Warsztat z promptowania (Google Colab) ma startować **od razu od klasyfikacji**, z pominięciem
etapów data collection / data preparation — na nie nie ma czasu na żywo. Wynika z tego twarde wymaganie:

- **gotowa, mała paczka komentarzy z etykietami musi istnieć przed warsztatem**,
- to ją (i tylko ją) zaciągają wszystkie kolejne notebooki/skrypty,
- ma uczyć **klasyfikacji multi-label** (jeden komentarz może mieć wiele aspektów),
- nie może składać się z samych najdłuższych „esejów" — uczestnicy nie mogą tracić czasu na czytanie.

Dotychczasowy `data/processed/workshop_sample.csv` (single-label, stare 8 kategorii) został **porzucony**.

## 2. Źródło danych

Recenzje gry *Dying Light: The Beast* ze Steam, podzielone na pozytywne i negatywne:

| Plik | Rekordów (parser CSV) | Uwaga |
|------|----------------------|-------|
| `data/raw/dying_light_beast_negative_full.csv` | **2 611** | wszystkie `language == english` |
| `data/raw/dying_light_beast_positive_sample.csv` | ~20 028 (linie) | nie użyty w golden dataset |

> Uwaga techniczna: `wc -l` na pliku negatywnym pokazuje ~14 675 „linii", ale recenzje zawierają
> wewnętrzne znaki nowej linii — realny licznik rekordów przez parser CSV to **2 611**.

Istotne kolumny: `review_id`, `review_text`, `language`, `votes_up` (liczba „podbić"/łapek —
sygnał wartości społecznościowej), `voted_up`, `playtime_hours`.

## 3. Decyzje doboru (ustalone z prowadzącym)

1. **Pula:** wyłącznie recenzje **negatywne**. Spójne z dotychczasowym codebookiem i taksonomią
   (zbudowanymi na skargach); taksonomia aspektów jest neutralna, więc i tak pasuje.
2. **Liczność:** **20** komentarzy (zredukowane z pierwotnych 25 — krótszy materiał na warsztacie).
3. **Strategia doboru:** *pokrycie aspektów + `votes_up` + limit długości* (a nie czyste „top podbić",
   bo te korelują z długością → byłyby same eseje).
4. **Zestaw etykiet:** **15 aspektów** wg `analysis/aspects_with_definitions.md`, z respektowaniem
   reguł granicznych „Granica (NIE tutaj)".
5. **Klasyfikujemy tylko NEGATYWNE aspekty** ([D-007]): etykietujemy wyłącznie to, co recenzent
   krytykuje; pochwał (np. „storytelling is the best") nie oznaczamy.

> **Ewolucja słownika ([D-008]):** w pierwszej wersji istniała oś ortogonalna `franchise_frame`
> (porównania do DL1/DL2). Została **usunięta** — werdykt „to powinno być DLC / to nie pełna gra"
> trafia teraz do `price`. Dodano kategorię **`gore`** (krew, miażdżone kości, system obrażeń ciała).
> Zestaw to nadal 15 etykiet.

## 4. Analiza eksploracyjna (uzasadnienie parametrów)

Na zbiorze 2 611 negatywnych (english) policzono rozkład długości i podbić:

- **Długość recenzji (znaki):** p25 = 107 · p50 = 337 · p75 = 847 · p90 = 2080.
  → Potwierdza ryzyko: górne percentyle to długie eseje.
- **Pas docelowy 200–700 znaków** (krótkie–średnie): **862 recenzje** w pasie.
- W pasie z `votes_up ≥ 3`: **158 recenzji** → materiału z nadmiarem na 20 pozycji.
- Kontrola jakości: top wg `votes_up` w pasie to recenzje merytoryczne i zwięzłe
  (np. `votes_up=144`, 291 znaków) — strategia działa.

**Wniosek:** pas **200–700 znaków**, ranking malejąco po `votes_up`, z dopuszczeniem rozszerzenia
do ~900 znaków tylko dla aspektów trudnych do pokrycia.

## 5. Pipeline selekcji

### Krok 1 — Pula kandydatów (zautomatyzowane, reprodukowalne)
Skrypt `scripts/select_golden.py` (read-only względem danych):
1. wczytuje plik negatywny, filtruje `language == english`,
2. zawęża do pasa **200–700 znaków**,
3. sortuje malejąco po `votes_up`,
4. zapisuje **top 150** do `analysis/golden_candidates.csv`
   (kolumny: `review_id, votes_up, len, review_text`; whitespace w tekście znormalizowany).

### Krok 2 — Ręczny dobór 20 (ekspercki, na bazie puli)
Z puli 150 wybrano 20 wg trzech reguł jednocześnie:
- **Pokrycie:** każdy z 15 aspektów ≥ 1 raz; `franchise_frame` ≥ 3.
- **Multi-label:** zdecydowana większość komentarzy ma ≥ 2 etykiety (cel dydaktyczny),
  z 2 celowo zostawionymi przykładami **single-label** (#16 performance, #17 world) jako proste przypadki.
- **Wartość:** w ramach powyższego preferowano wyższe `votes_up`.

Rozkład `votes_up` wybranych 20: **144, 131, 119, 99, 86, 70, 51, 31, 28, 27, 23, 21, 21, 19,
18, 16, 13, 11, 8, 3** — czyli niemal cały szczyt puli; jedyny niski (`3`) dociągnięto, by pokryć `coop`.

### Krok 3 — Ręczne etykietowanie multi-label
Każdy komentarz oznaczono ręcznie wg taksonomii, **tylko dla aspektów krytykowanych** ([D-007]).
Kluczowe rozstrzygnięcia granic:
- **Negative-only:** jeśli aspekt jest chwalony (np. „parkour improved", „music was good"), etykiety NIE nadajemy.
- **`combat` vs `enemies`** przy „zombie grab/grapple": gdy ocena dotyczy feelu/przerwania walki → `combat`,
  a gdy częstotliwości/mechaniki/balansu wrogów → `enemies`; często **oba naraz**.
- **`bugs` vs `performance`:** spadki FPS/optymalizacja → `performance`; crashe/glitche/softlocki → `bugs`.
- **`content` vs `price`:** ilość treści/długość/powtarzalność → `content`; wartość za pieniądze, a także
  werdykt „to powinno być DLC / to nie pełna gra" → `price`.
- **`world` vs `graphics`:** projekt mapy/eksploracja/fast travel/pojazd → `world`; warstwa wizualna → `graphics`.
- **`gore` vs `graphics`/`combat`:** krew, rozczłonkowanie, miażdżenie kości, system obrażeń ciała → `gore`;
  ogólny art direction → `graphics`; feel/impact i balans walki → `combat`.

Wynik jako plik do recenzji: `analysis/golden_proposed.md` (pełne teksty + `LABELS` + uzasadnienie).

### Krok 3a — Rewizja po uwagach prowadzącego
Po pierwszym przeglądzie wprowadzono [D-007] (negative-only) i [D-008] (−`franchise_frame`, +`gore`).
Reguła negative-only obniżyła pokrycie `night_horror` i `audio` do 0 (były wcześniej ujęte jako pochwały),
więc **podmieniono 2 komentarze** (z tej samej puli 150): usunięto `209560256` (dup parkour/world) oraz
`204629841` (dup performance); dodano `207836299` (negatywny night_horror) i `215810589` (negatywny voice acting → audio).

## 6. Wynik — pokrycie aspektów (20 komentarzy)

| Aspekt | # | | Aspekt | # |
|--------|---|---|--------|---|
| combat | 10 | | performance | 2 |
| parkour | 4 | | graphics | 2 |
| enemies | 5 | | audio | 1 |
| night_horror | 1 | | content | 8 |
| progression | 6 | | price | 9 |
| world | 6 | | coop | 1 |
| story | 8 | | gore | 2 |
| bugs | 3 | | | |

**Znane ograniczenia pokrycia:** `night_horror`, `audio` i `coop` mają po 1 przykładzie — w recenzjach
**negatywnych** (a tym bardziej po regule negative-only) występują rzadko. Świadomy kompromis;
ewentualne dociągnięcie drugiego przykładu wymagałoby rozluźnienia limitu długości lub zejścia z `votes_up`.

## 7. Artefakty procesu

| Plik | Rola | Etap |
|------|------|------|
| `scripts/select_golden.py` | generator puli kandydatów (reprodukowalny) | 1 |
| `analysis/aspects_with_definitions.md` | taksonomia 15 aspektów + `franchise_frame` (źródło prawdy) | wejście |
| `analysis/golden_candidates.csv` | top 150 kandydatów w pasie długości | 1 |
| `analysis/golden_proposed.md` | 20 komentarzy + proponowane etykiety + uzasadnienia (do weryfikacji) | 2–3 |
| `data/evaluation/golden_dataset.{json,csv}` | **finalna paczka** zaciągana przez notebooki | finalizacja |

Schemat rekordu finalnego: `{ id, review_id, votes_up, review_text, labels: [...] }`
(w CSV `labels` rozdzielone `;`). Walidacja: 20 rekordów, brak duplikatów `review_id`,
wszystkie etykiety w dozwolonym zbiorze 15 kategorii (`franchise_frame` porzucony — [D-008]).

## 8. Reprodukcja

```bash
python3 scripts/select_golden.py   # -> analysis/golden_candidates.csv (150 wierszy)
# ręczny dobór 20 + etykietowanie -> analysis/golden_proposed.md
# po akceptacji etykiet -> data/evaluation/golden_dataset.{json,csv}
```

## 9. Status i następne kroki

- [x] EDA, pula kandydatów, dobór 20, wstępne etykiety (`golden_proposed.md`).
- [x] **Weryfikacja etykiet przez prowadzącego** (punkt wstrzymania zdjęty — [D-013]).
- [x] Finalizacja `data/evaluation/golden_dataset.{json,csv}` — skrypt `scripts/build_golden_dataset.py` (T1, 2026-06-20). 20 rekordów, 15 kategorii, walidacja i pokrycie OK.
- [ ] Przepisanie notebooków 03–06 na taksonomię 15+1 i ładowanie golden datasetu; odchudzenie skryptów ewaluacji.

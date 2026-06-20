# TO-DO — przebudowa notebooków warsztatowych

> Plan wykonawczy wynikający z decyzji **D-009…D-013** (`docs/decisions_backlog.md`).
> Zadania są tak rozbite, by każde dało się zrealizować w **osobnym, czystym oknie kontekstowym**.
> Każde zadanie ma: cel, pliki, kroki, kryteria akceptacji, zależności.

## Status zadań

| Zadanie | Status |
|---------|--------|
| **T1** — golden dataset | ✅ **DONE** (2026-06-20) |
| **T2** — `workshop_utils.py` | ✅ **DONE** (2026-06-20) |
| **T3** — notebook 03 (szablon) | ✅ **DONE** (2026-06-20) |
| **T4** — notebook 04 | ✅ **DONE** (2026-06-20) |
| **T5** — notebook 05 | ✅ **DONE** (2026-06-20) |
| **T6** — notebook 06 | ✅ **DONE** (2026-06-20) |
| T7 — trym repo | ☐ OPEN |

## Mapa zależności

```
T1 (golden dataset)  ─┐
                      ├─► T3 (notebook 03 = szablon) ─► T4 (04) ┐
T2 (workshop_utils) ──┘                              ├─► T5 (05) ├─► T7 (trym repo) [OSTATNIE]
                                                     └─► T6 (06) ┘
T1 ∥ T2  (równolegle, wg wspólnego SCHEMATU poniżej)
T4 ∥ T5 ∥ T6  (równolegle, po ukończeniu T3)
requirements.txt (część T7) — można zrobić w dowolnym momencie; sam trym main = OSTATNI
```

- **Równolegle od startu:** T1 i T2 (oba korzystają ze wspólnego SCHEMATU danych zdefiniowanego niżej).
- **Po T1+T2:** T3 (ustala wspólny szablon notebooka).
- **Po T3:** T4, T5, T6 — równolegle względem siebie.
- **Na końcu, po T1–T6:** T7 (trym repozytorium).

---

## SCHEMAT WSPÓLNY (kontrakt między zadaniami — przeczytaj przed T1/T2/T3)

**`data/evaluation/golden_dataset.json`** — lista 20 obiektów:
```json
{
  "id": 1,
  "review_id": "205490944",
  "votes_up": 144,
  "review_text": "I very much enjoyed the 30+ hours...",
  "labels": ["content", "price"]
}
```
- `data/evaluation/golden_dataset.csv` — te same kolumny; `labels` jako string rozdzielony `;`.
- **15 kategorii (kody):** `combat, parkour, enemies, night_horror, progression, world, story, bugs, performance, graphics, audio, content, price, coop, gore`. Źródło definicji: `analysis/aspects_with_definitions.md`.
- **Helper API** (`notebooks/workshop_utils.py`) udostępnia: `CATEGORIES` (dict kod→definicja), `MODEL`, `create_workshop_client(project_id, location="europe-west4")`, `load_golden()`, `show_categories()`, `show_golden_reviews()`, `MatchStrategy`, `evaluate_trial(...)`, `compare_trials(...)`.
- **Model:** `google/gemini-2.5-flash-lite`, `temperature=0`, bez reasoningu.
- **Import helpera w notebooku:** helper leży w `notebooks/`. W setupie notebooka trzeba dodać katalog `notebooks/` do `sys.path` (w Colab po `git clone`: `sys.path.insert(0, f"{REPO_DIR}/notebooks")`; lokalnie cwd zwykle = `notebooks/`), żeby `from workshop_utils import ...` działało. To NIE jest import z `src/`.
- **Tryb instructor:** Gemini przez endpoint Vertex OpenAI-compat wymaga sprawdzonego trybu — użyj `instructor.patch(client, mode=instructor.Mode.MD_JSON)` (tak jak działający `src/utils/llm_client.py`). Domyślny tryb `TOOLS` może nie zadziałać na tym endpoincie.

---

## T1 — Finalny golden dataset (skrypt + dane) ✅ DONE (2026-06-20)

> **Zrealizowane.** `scripts/build_golden_dataset.py` generuje finalny
> `data/evaluation/golden_dataset.{json,csv}` (20 rec./15 kat.). Stara wersja
> (100 rec./8 kat.) nadpisana. Walidacja i rozkład pokrycia zgodne z kryteriami
> akceptacji (combat 10, price 9, content 8, story 8, progression 6, world 6,
> enemies 5, parkour 4, bugs 3, performance 2, graphics 2, gore 2,
> night_horror 1, audio 1, coop 1). JSON↔CSV spójne.

- **Zależności:** brak (równolegle z T2).
- **Cel:** wygenerować finalny `data/evaluation/golden_dataset.{json,csv}` (20 rec./15 kat.) i zastąpić starą wersję (100 rec./8 kat.).
- **Pliki:** nowy `scripts/build_golden_dataset.py`; nadpisane `data/evaluation/golden_dataset.json`; nowy `data/evaluation/golden_dataset.csv`.
- **Kroki:**
  1. W skrypcie zaszyj mapę `review_id → labels` dla 20 recenzji **dokładnie wg `analysis/golden_proposed.md`** (sekcje 1–20; pola „LABELS"). Lista review_id: 205490944, 204662234, 204658386, 204958301, 217659097, 204834661, 204957991, 204639308, 204632770, 204870501, 206740572, 204878716, 204642036, 205076487, 205100534, 204641037, 204926978, 205334075, 207836299, 215810589.
  2. Wczytaj pełne `review_text` (i `votes_up`) z `analysis/golden_candidates.csv` po `review_id` (NIE używaj skróconych tekstów z `.md`).
  3. Walidacja (twarde asercje): dokładnie 20 rekordów; brak duplikatów `review_id`; każda etykieta ∈ 15 kategorii; każdy `review_id` znaleziony w źródle.
  4. Zapis `json` (`indent=2, ensure_ascii=False`) i `csv` (labels łączone `;`).
- **Akceptacja:** `python scripts/build_golden_dataset.py` kończy się bez błędu; `len(json)==20`; zbiór etykiet ⊆ 15 kategorii; `csv` i `json` mają zgodne `review_id` i etykiety. Rozkład pokrycia aspektów zgodny z nagłówkiem `golden_proposed.md` (combat 10, price 9, content 8, story 8, progression 6, world 6, enemies 5, parkour 4, bugs 3, performance 2, graphics 2, gore 2, night_horror 1, audio 1, coop 1).

## T2 — Helper `notebooks/workshop_utils.py`

- **Zależności:** brak (równolegle z T1; trzymaj się SCHEMATU). Integracyjny test po T1.
- **Cel:** jeden samowystarczalny plik (BEZ importów z `src/`) z setupem LLM (OAuth), prezentacją danych „pod ręką" i ewaluacją z czułością/swoistością.
- **Pliki:** nowy `notebooks/workshop_utils.py`.
- **Kroki:**
  1. **`CATEGORIES`** — dict 15 kodów → definicja (przepisz z `analysis/aspects_with_definitions.md`). **`MODEL = "google/gemini-2.5-flash-lite"`**.
  2. **`create_workshop_client(project_id, location="europe-west4")`** — pozyskaj token OAuth jak w `notebooks/00_smoke_test_auth.ipynb` (próbuj `google.colab.auth.authenticate_user()`, fallback `google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])` + `creds.refresh(...)`); zbuduj `BASE_URL = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi"` (BEZ `/chat/completions` — SDK dokleja sam); zwróć `instructor.patch(OpenAI(api_key=token, base_url=BASE_URL), mode=instructor.Mode.MD_JSON)`. **Bez** custom httpx, **bez** `x-goog-api-key`. Token Bearer idzie naturalnie przez OpenAI SDK. (Uwaga: token OAuth wygasa ~1h — przy długiej sesji może wymagać ponownego `create_workshop_client`.)
  3. **`load_golden()`** — wczytaj `data/evaluation/golden_dataset.json` (ścieżka odporna na Colab/lokalnie); zwróć rekordy + wygodne `texts`, `labels`.
  4. **`show_categories()`** i **`show_golden_reviews()`** — kolapsowalne tabele HTML (`IPython.display.HTML` + `<details>`), fallback `print`. `show_golden_reviews()` pokazuje tekst recenzji + przypisane etykiety (żeby nie wracać do innych plików).
  5. **Ewaluacja** — przenieś z `src/evaluation/evaluator.py`: `MatchStrategy`, `evaluate_trial`, `compare_trials`, modele wyników. **Dodaj** liczenie confusion matrix multi-label względem uniwersum 15 kategorii: per recenzja `TP=|E∩P|, FN=|E\P|, FP=|P\E|, TN=|U\(E∪P)|`. Policz **micro** czułość `ΣTP/(ΣTP+ΣFN)` i swoistość `ΣTN/(ΣTN+ΣFP)` oraz **per-kategoria**. Rozszerz `TrialResult.display()` o: accuracy + czułość + swoistość (micro) + tabelę per-kategoria.
- **Akceptacja:** `python -c "import ast; ast.parse(open('notebooks/workshop_utils.py').read())"` OK; `grep -n "src\." notebooks/workshop_utils.py` pusty; ręczny mini-test `evaluate_trial` na sztucznych predykcjach daje czułość/swoistość zgodne z ręcznym wyliczeniem; tabela per-kategoria ma 15 wierszy.

## T3 — Notebook 03 (zero-shot) = wzorcowy szablon ✅ DONE (2026-06-20)

> **Zrealizowane.** `notebooks/03_iteration1_basic_prompting.ipynb` przebudowany
> na wzorcowy szablon (13 komórek wg kolejności z kroków): tytuł+cel (bez
> czas/poziom, bez „Anatomii promptu") → setup OAuth (`create_workshop_client`,
> zero kluczy API, `sys.path` na `notebooks/`) → blok „mamy ekspercką
> klasyfikację" (`show_categories()` + `show_golden_reviews()` + `load_golden()`)
> → definicja zero-shot (bez zalet/wad) → zadanie → puste `SYSTEM_PROMPT=""` /
> `USER_PROMPT=""` → `classify_review` (surowy string + parsowanie) + test na 1
> recenzji → klasyfikacja 20 (`tqdm`) → kanoniczny opis metryk
> (accuracy/czułość/swoistość, micro + per-kategoria) → `evaluate_trial().display()`
> → podsumowanie/pomost do iteracji 2. Weryfikacja: `nbconvert --to script` bez
> błędów; brak `src.`; brak referencji do kluczy API; puste prompty obecne.
> **Do potwierdzenia na realnym OAuth:** czy `client.chat.completions.create(...)`
> bez `response_model` działa na kliencie po `instructor.patch` (MD_JSON).
> Ten szablon jest wzorcem dla T4–T6.

- **Zależności:** T1 + T2.
- **Cel:** przebudować `03_iteration1_basic_prompting.ipynb` wg D-011; ustala szablon dla 04–06.
- **Pliki:** `notebooks/03_iteration1_basic_prompting.ipynb`.
- **Kroki (kolejność komórek):**
  1. Markdown: tytuł + krótki cel. **Usuń** „czas trwania / poziom" i sekcję „Anatomia promptu".
  2. Markdown + code: **Setup** — minimalna instalacja / `git clone`, widoczna konfiguracja (`PROJECT_ID=""`, `LOCATION="europe-west4"`, `MODEL`), `client = create_workshop_client(PROJECT_ID)`. **OAuth, zero kluczy API.** Usuń stare komórki z `VERTEX_AI_API_KEY` / `create_client` / `LLMProvider`.
  3. Markdown: „Mamy gotową ekspercką klasyfikację 20 recenzji — Twoim zadaniem jest ją **odtworzyć**, nie wymyślać od zera." + code: `show_categories()` i `show_golden_reviews()` (kolapsowalne). (W 03 ten blok jest **po** setupie.)
  4. Markdown: **definicja zero-shot** (1–3 zdania). **Bez** list zalet/wad.
  5. Markdown: **Zadanie** — opisz co zrobić (napisz prompt klasyfikujący 20 recenzji do 15 kategorii, multi-label), **bez** podpowiedzi co wpisać w prompt.
  6. Code: **puste** `SYSTEM_PROMPT = ""` i `USER_PROMPT = ""`.
  7. Code: funkcja klasyfikująca (surowy string + ręczne parsowanie do listy etykiet) + test na 1 recenzji.
  8. Code: klasyfikacja wszystkich 20 recenzji (`tqdm`).
  9. Markdown: **wyjaśnienie metryk** — accuracy, czułość (sensitivity), swoistość (specificity) — pełny opis (to „kanoniczna" komórka, w 04–06 tylko skrót).
  10. Code: `evaluate_trial(...).display()` (jeden model) — accuracy + czułość + swoistość + per-kategoria.
  11. Markdown: krótkie podsumowanie / pomost do iteracji 2 (bez zdradzania techniki naprzód).
- **Akceptacja:** `jupyter nbconvert --to script` bez błędów; brak importów z `src.`; brak referencji do kluczy API; puste prompty obecne; po ręcznym uzupełnieniu promptów pipeline przechodzi i `display()` pokazuje 3 metryki + tabelę per-kategoria.

## T4 — Notebook 04 (structured output / Pydantic + Instructor) ✅ DONE (2026-06-20)

> **Zrealizowane.** `notebooks/04_iteration2_structured_output.ipynb` przebudowany
> wg szablonu T3 (14 komórek): tytuł+cel (bez czas/poziom) → setup OAuth
> (`create_workshop_client`, zero kluczy API, `sys.path` na `notebooks/`) → blok
> „mamy ekspercką klasyfikację" na górze → definicja structured output (bez
> zalet/wad) → zadanie → minimalny model Pydantic
> (`ReviewClassification(categories: list[str])`) + puste `SYSTEM_PROMPT`/`USER_PROMPT`
> → `classify_review` z `response_model=` (zwraca obiekt Pydantic) + test na 1
> recenzji → klasyfikacja 20 (`tqdm`, `result.categories`) → skrót metryk z
> odwołaniem do 03 → `evaluate_trial().display()` → podsumowanie/pomost do iteracji 3.
> Weryfikacja: `nbconvert --to script` + `ast.parse` OK; brak `src.`; brak kluczy
> API; puste prompty obecne; klasyfikacja zwraca Pydantic, nie string.

- **Zależności:** T1 + T2 + T3 (szablon). Równolegle z T5, T6.
- **Cel:** przebudować `04_iteration2_structured_output.ipynb` wg szablonu z T3.
- **Pliki:** `notebooks/04_iteration2_structured_output.ipynb`.
- **Kroki:** zastosuj szablon T3; **blok „mamy klasyfikację" przenieś na górę** (po cel/setup). Mechanizm iteracji: wprowadź `instructor` + **minimalny** model Pydantic (`class ReviewClassification(BaseModel): categories: list[str]`) i `response_model=...`. **Prompty pozostają puste.** Definicja techniki bez zalet/wad. Metryki — skrót + odwołanie do opisu z 03.
- **Akceptacja:** jak T3; dodatkowo klasyfikacja zwraca obiekt Pydantic (nie string); brak gotowych promptów.

## T5 — Notebook 05 (chain-of-thought) ✅ DONE (2026-06-20)

> **Zrealizowane.** `notebooks/05_iteration3_chain_of_thought.ipynb` przebudowany
> wg szablonu T3 (15 komórek): tytuł+cel (bez czas/poziom, bez wykładu o teorii) →
> setup OAuth (`create_workshop_client`, zero kluczy API, import `pydantic`) → blok
> „mamy ekspercką klasyfikację" na górze (`show_categories` + `show_golden_reviews`
> + `load_golden`) → krótka definicja CoT (bez zalet/wad) → zadanie naprowadzające
> na ideę: uczestnik sam dodaje pole `reasoning` PRZED `categories` w modelu
> Pydantic (placeholder `class ReviewClassification(BaseModel): ...` z TODO, BEZ
> gotowego modelu) → puste `SYSTEM_PROMPT=""`/`USER_PROMPT=""` → `classify_review`
> z `response_model=ReviewClassification` (obiekt Pydantic, nie string) + test na 1
> recenzji → klasyfikacja 20 (`tqdm`) → metryki: skrót + odwołanie do notebooka 03
> → `evaluate_trial().display()` → podsumowanie/pomost do few-shot. Weryfikacja:
> `nbconvert --to script` + `ast.parse` OK; brak `src.`; brak kluczy API; puste
> prompty obecne; placeholder reasoning bez gotowego rozwiązania.

- **Zależności:** T1 + T2 + T3. Równolegle z T4, T6.
- **Cel:** przebudować `05_iteration3_chain_of_thought.ipynb` wg szablonu.
- **Pliki:** `notebooks/05_iteration3_chain_of_thought.ipynb`.
- **Kroki:** szablon T3 + blok danych na górze. Krótka definicja CoT (bez zalet/wad). **Zadanie:** uczestnik ma **sam** wymusić rozumowanie przez dodanie pola `reasoning` (PRZED `categories`) w modelu Pydantic — naprowadź na ideę w treści zadania, **nie** podawaj gotowego modelu ani promptu. Prompty puste. (To realizuje „wymuszenie CoT w modelach Pydantic" z D-011.)
- **Akceptacja:** jak T3; brak gotowego CoT-promptu/modelu z polem reasoning podanego wprost (jest placeholder do uzupełnienia przez uczestnika).

## T6 — Notebook 06 (few-shot) ✅ DONE (2026-06-20)

> **Zrealizowane.** `notebooks/06_iteration4_few_shot.ipynb` przebudowany wg
> szablonu T3 (15 komórek): tytuł+cel (bez czas/poziom, bez wykładowej „Teorii”) →
> setup OAuth (`create_workshop_client`, zero kluczy API, `sys.path` na `notebooks/`,
> import `pydantic`) → blok „mamy ekspercką klasyfikację” na górze
> (`show_categories` + `show_golden_reviews` + `load_golden`) → krótka definicja
> few-shot (bez zalet/wad) → zadanie naprowadzające: uczestnik sam dobiera przykłady
> (`FEW_SHOT_EXAMPLES = []`, pusty placeholder + opis formatu, BEZ gotowych
> przykładów) i wplata je w puste `SYSTEM_PROMPT=""`/`USER_PROMPT=""` → minimalny
> model `ReviewClassification(categories: list[str])` → `classify_review` z
> `response_model=` (obiekt Pydantic, nie string) + test na 1 recenzji →
> klasyfikacja 20 (`tqdm`) → metryki: skrót + odwołanie do notebooka 03 →
> `evaluate_trial().display()` (jedno-modelowo, bez `compare_trials`) →
> podsumowanie zamykające cykl 4 technik. Weryfikacja: `nbconvert --to script` +
> `ast.parse` OK; brak `src.`; brak kluczy API; puste prompty; `FEW_SHOT_EXAMPLES`
> pusty bez gotowych przykładów.

- **Zależności:** T1 + T2 + T3. Równolegle z T4, T5.
- **Cel:** przebudować `06_iteration4_few_shot.ipynb` wg szablonu.
- **Pliki:** `notebooks/06_iteration4_few_shot.ipynb`.
- **Kroki:** szablon T3 + blok danych na górze. Krótka definicja few-shot (bez zalet/wad). **Zadanie:** uczestnik dobiera przykłady — `FEW_SHOT_EXAMPLES = []` (pusty placeholder), bez gotowych przykładów. Prompty puste. Możesz zostawić zwięzłą końcową komórkę porównawczą (`compare_trials`) jeśli pasuje do jednego-modelowej ewaluacji, ale bez wstawiania gotowych wyników.
- **Akceptacja:** jak T3; `FEW_SHOT_EXAMPLES` pusty; brak gotowych przykładów/promptów.

## T7 — Trym repozytorium + slim requirements (OSTATNIE)

- **Zależności:** T1–T6 ukończone i przetestowane. (Sam `requirements.txt` można odchudzić wcześniej.)
- **Cel:** `main` klonowany przez uczestników = minimum; materiały autorskie zachowane.
- **Pliki/operacje:**
  1. `requirements.txt` → tylko: `openai, instructor, pydantic, google-auth, pandas, matplotlib, tqdm`.
  2. Utwórz gałąź `authoring` z pełną zawartością (backup historii).
  3. Na `main` **usuń z klonowanej zawartości:** `src/`, `scripts/scrape_reviews.py`, `scripts/prepare_workshop_data.py`, `scripts/test_pipeline.py`, `analysis/`, autorskie `docs/`, `data/raw`, `data/processed` (w tym porzucony `workshop_sample.csv`), duże PDF-y (`Program warsztatowy Techland.pdf`).
  4. **Notebooki autorskie** `01_data_collection.ipynb` i `02_data_preparation.ipynb` → przenieś na `authoring`, usuń z `main` (warsztat startuje od razu od klasyfikacji — D-006). **Zostaw na `main`:** `00_smoke_test_auth.ipynb` (uczestnicy testują swoje środowisko), `03`–`06`, `workshop_utils.py`.
  5. **Zostaw na `main`:** `notebooks/` (00, 03–06 + `workshop_utils.py`), `data/evaluation/golden_dataset.{json,csv}`, `requirements.txt`, skrócony `README`/`QUICKSTART`, `.gitignore`.
  6. Zaktualizuj `.gitignore` i krótkie `README`/`QUICKSTART` pod nowy, minimalny przepływ.
  7. **Uwaga:** smoke test ([D-004]) sprawdza obecność katalogu `data` — po trymie `data/evaluation/` musi istnieć (golden dataset go zapewnia). Zweryfikuj sanity-check w `00_*` po trymie.
- **Akceptacja:** `git clone --depth 1` `main` daje tylko notatniki + dane + requirements; `00_smoke_test_auth.ipynb` (sanity check `requirements.txt`/`notebooks`/`data`) zielony; notebooki 03–06 działają z czystego klona (import `workshop_utils` z katalogu `notebooks/`).

---

## Weryfikacja końcowa (po T1–T7)

1. Build danych: `python scripts/build_golden_dataset.py` → 20 rekordów, etykiety ⊆ 15 kategorii.
2. Składnia: `jupyter nbconvert --to script` na 03–06 + `ast.parse` na `workshop_utils.py` — bez błędów, bez `src.`.
3. Realny OAuth (Colab lub `gcloud auth application-default login`): `create_workshop_client(PROJECT_ID)` + klasyfikacja 1 recenzji → 200, brak reasoningu.
4. Przejście notebooka 03 „jak uczestnik": puste prompty → klasyfikacja 20 → `display()` = accuracy + czułość + swoistość + per-kategoria.
5. Czysty klon przyciętego `main` → notatniki uruchamialne, smoke test zielony.

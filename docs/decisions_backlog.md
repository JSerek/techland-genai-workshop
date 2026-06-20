# Decisions Backlog

> **Append-only.** Nowe decyzje i zmiany statusu dopisujemy na dole, nie usuwamy i nie nadpisujemy istniejących wpisów. Jeśli decyzja się zmienia — dodajemy nowy wpis i zmieniamy status starego na `SUPERSEDED` z linkiem do nowego ID.

**Statusy:** `OPEN` (do zrobienia / do ustalenia) · `IN PROGRESS` · `DONE` · `DEFERRED` (świadomie odłożone) · `BLOCKED` · `SUPERSEDED`

---

## D-001 — Sonda do testu autoryzacji Vertex AI (API key + OAuth)

- **Status:** SUPERSEDED → [D-004]
- **Zaktualizowano:** 2026-06-20 14:10 CEST
- **Kontekst:** Uczestnicy szkolenia będą korzystać z firmowego środowiska GCP, nie z konta prowadzącego (`projectgca`). Nie wiemy, czy firma używa Vertex AI Express (klucz API + nagłówek `x-goog-api-key`), czy standardowego Vertex AI (OAuth / Bearer). To główna blokada techniczna całego warsztatu. W poniedziałek (2026-06-22) spotkanie z technicznym uczestnikiem, który pomoże przetestować ich środowisko.
- **Decyzja:** Tworzymy minimalny notebook-sondę `notebooks/00_smoke_test_auth.ipynb` testujący połączenie z Gemini przez Vertex AI w **obu** trybach autoryzacji naraz (API key oraz OAuth), żeby na jednym spotkaniu sprawdzić, który zadziała na firmowym GCP. Sonda używa surowego `requests` (przejrzyste statusy/nagłówki do debugowania w korporacyjnym środowisku).

---

## D-002 — Ścieżka OAuth/Bearer w `llm_client.py` dla standardowego Vertex AI

- **Status:** DEFERRED
- **Zaktualizowano:** 2026-06-20 13:49 CEST
- **Kontekst:** Obecny `_create_vertex_ai_client` podmienia nagłówek na `x-goog-api-key`, co działa tylko z Vertex AI Express. Standardowy firmowy Vertex AI używa OAuth (`Authorization: Bearer`), gdzie podmiana nagłówka psuje request. Trzeba dorobić wariant Bearer, ale dopiero po ustaleniu, czego faktycznie używa firma (patrz [D-001]).
- **Decyzja:** Odłożone do czasu wyniku sondy z [D-001]. Po potwierdzeniu modelu auth dorobimy w `llm_client.py` gałąź OAuth/Bearer (token z `google.auth` / ADC), tak aby pokryć oba scenariusze.

---

## D-003 — Naprawa bugów w `notebooks/03_iteration1_basic_prompting.ipynb`

- **Status:** DONE (rozwiązane przez T3)
- **Zaktualizowano:** 2026-06-20 13:49 CEST
- **Kontekst:** Przy przeglądzie notebooka 03 znaleziono błędy, które wywrócą wykonanie w Colab:
  1. cell 15: użyto `review_texts[0]` — zmienna nie istnieje, poprawna to `golden_texts`.
  2. cell 10: ponowne ładowanie golden dataset ze **względnej** ścieżki `../data/...`, sprzeczne z `DATA_DIR` (cell 7) opartym o `REPO_DIR/data`; w Colab `../data` nie istnieje.
  3. Do weryfikacji: czy `client.chat.completions.create(...)` (surowy call OpenAI) działa na obiekcie zwróconym przez `instructor.patch` — w iteracji 1 nie używamy structured output.
- **Decyzja:** Do naprawienia przy dopracowywaniu treści notebooków. Na razie zarejestrowane, nie ruszamy kodu.
- **Aktualizacja 2026-06-20 (T3 wykonane):** Notebook 03 przebudowany od zera wg [D-011], więc wszystkie trzy bugi zniknęły: (1) test klasyfikacji używa `golden.texts[0]` (helper), nie `review_texts`; (2) golden ładowany raz przez `load_golden()` (ścieżka odporna na Colab/lokalnie), bez `../data`; (3) punkt do weryfikacji — `client.chat.completions.create(...)` bez `response_model` na kliencie po `instructor.patch` (MD_JSON) — pozostaje do potwierdzenia na realnym OAuth (punkt 3 weryfikacji końcowej w `docs/to_do.md`).

---

## D-004 — Zakres sondy: tylko OAuth, region europe-west4, + test git clone

- **Status:** DONE
- **Zaktualizowano:** 2026-06-20 14:10 CEST
- **Kontekst:** Prowadzący samodzielnie przetestował GCP innej firmy i ustalił, że firmowy Vertex AI **nie udostępnia kluczy API** — autoryzacja idzie wyłącznie przez OAuth. Tryb A (API key / `x-goog-api-key`) z [D-001] jest więc nieistotny. Dodatkowo region warsztatu to **`europe-west4`**, a przed poniedziałkowym (2026-06-22) spotkaniem z technicznym uczestnikiem trzeba potwierdzić również, że z firmowego środowiska da się sklonować repo z GitHub.
- **Decyzja:** Zastępuje [D-001]. Sonda `notebooks/00_smoke_test_auth.ipynb`:
  1. **usuwa** ścieżkę API key — testuje wyłącznie OAuth (Bearer),
  2. ustawia `LOCATION = "europe-west4"`,
  3. dodaje **test `git clone`** repo warsztatowego (`--depth 1` do katalogu tymczasowego + sanity check obecności `requirements.txt`, `notebooks`, `data`),
  4. podsumowanie raportuje oba wyniki: Vertex AI (OAuth) oraz git clone.
- **Następstwa:** potwierdza kierunek [D-002] (potrzebna gałąź OAuth/Bearer w `llm_client.py`, bo API key odpada). Implikacja do późniejszego sprawdzenia: notebooki warsztatowe (np. 03) używają sekretów `VERTEX_AI_API_KEY`/`x-goog-api-key` — do przerobienia na OAuth po potwierdzeniu sondą.

---

## D-005 — Higiena sekretów: GitHub PAT wyjęty z konfiguracji

- **Status:** DONE
- **Zaktualizowano:** 2026-06-20 14:35 CEST
- **Kontekst:** GitHub PAT (uprawnienia `push`+`admin` do repo) był zaszyty jawnym tekstem w `git remote origin`. Przy próbie bezpiecznego zapisania nowego tokena trafił dodatkowo do historii shella (`~/.zsh_history`). Token z prawem zapisu = realne ryzyko przejęcia repo przy wycieku.
- **Decyzja:**
  1. Usunięto token z `git remote` → czysty URL `https://github.com/JSerek/techland-genai-workshop.git`.
  2. Poświadczenia trzymane w **macOS Keychain** (`credential.helper = osxkeychain`); `git push` autoryzuje bez wklejania tokena do URL/plików.
  3. Stary, potencjalnie wyciekły token odwołany na GitHub i wygenerowany nowy.
  4. Token usunięty z `~/.zsh_history` (`sed` + `fc -W/-R`); zweryfikowano brak wystąpień `github_pat_`.
- **Lekcja na przyszłość:** nigdy nie wklejać sekretów do URL-i, plików w repo ani komend shella (lądują w historii). Token podawać wyłącznie przez interaktywny monit `git push` (input nie jest echo'wany) lub menedżer haseł.

---

## D-006 — Golden dataset: 20 negatywnych recenzji, multi-label, taksonomia 15+1

- **Status:** IN PROGRESS
- **Zaktualizowano:** 2026-06-20 15:40 CEST
- **Kontekst:** Warsztat startuje od razu od klasyfikacji (bez data collection/preparation na żywo), więc gotowa paczka komentarzy z etykietami musi istnieć wcześniej i być jedynym źródłem dla kolejnych notebooków. Dotychczasowy `data/processed/workshop_sample.csv` (single-label, stare 8 kategorii) jest niewystarczający. Autorytatywna klasyfikacja to `analysis/aspects_with_definitions.md` (15 aspektów + `franchise_frame`).
- **Decyzja:** Zbudować golden dataset wg ustaleń: pula **tylko negatywne** (english, 2 611 rekordów), **20** komentarzy, dobór *pokrycie aspektów + `votes_up` + limit długości 200–700 znaków* (świadomie NIE „top podbić", bo to korelacja z długością), etykietowanie **multi-label** względem pełnych 15 aspektów + `franchise_frame`. Pipeline: `scripts/select_golden.py` → `analysis/golden_candidates.csv` (top 150) → ręczny dobór i etykiety → `analysis/golden_proposed.md` → po akceptacji `data/evaluation/golden_dataset.{json,csv}`.
- **Pełny opis procesu analitycznego:** `docs/golden_dataset_methodology.md`.
- **Następstwa / do zrobienia:** (1) weryfikacja etykiet przez prowadzącego (punkt wstrzymania); (2) finalizacja `golden_dataset.{json,csv}`; (3) przepisanie notebooków 03–06 na taksonomię i ładowanie golden datasetu — łączy się z [D-003].
- **Aktualizacja 2026-06-20 16:10 CEST:** etykiety zrewidowane po uwagach prowadzącego — patrz [D-007] (negative-only) i [D-008] (−`franchise_frame`, +`gore`). Wciąż oczekują finalnej akceptacji przed generacją `golden_dataset.{json,csv}`.

---

## D-007 — Zasada warsztatu: klasyfikujemy tylko NEGATYWNE aspekty

- **Status:** DONE
- **Zaktualizowano:** 2026-06-20 16:10 CEST
- **Kontekst:** W pierwszej wersji etykiet wyciągano każdy wspomniany aspekt, także te chwalone (np. „storytelling is the best", „parkour improved"). Warsztat dotyczy analizy recenzji **negatywnych**, więc interesują nas wyłącznie aspekty **krytykowane** — etykietowanie pochwał zaszumia golden dataset i myli model.
- **Decyzja:** Reguła obowiązująca dla całego warsztatu: w recenzji negatywnej oznaczamy **tylko te aspekty, które recenzent krytykuje**. Pochwał nie etykietujemy.
- **Następstwa:** Po zastosowaniu reguły `night_horror` i `audio` spadły do 0 przykładów (były ujęte pozytywnie). Decyzja prowadzącego: **podmienić 2 komentarze** z tej samej puli 150 — usunięto `209560256` (dup parkour/world) i `204629841` (dup performance); dodano `207836299` (negatywny night_horror) oraz `215810589` (negatywny voice acting → audio). Pełne pokrycie 15 kategorii utrzymane (night_horror/audio/coop po 1).

---

## D-008 — Słownik aspektów: usunięcie `franchise_frame`, dodanie `gore`

- **Status:** DONE
- **Zaktualizowano:** 2026-06-20 16:10 CEST
- **Kontekst:** (1) Oś `franchise_frame` (porównania do DL1/DL2, „to powinno być DLC") okazała się zbędna jako osobna kategoria — werdykt „to powinno być DLC / to nie pełna gra / dodatek za cenę pełnej gry" jest w istocie oceną **wartości za pieniądze**. (2) W tej grze istotnym, wyróżnionym aspektem są efekty gore (krew, miażdżone kości, system obrażeń), których nie obejmował czysto wizualny `graphics`.
- **Decyzja:** W `analysis/aspects_with_definitions.md`: (a) **usunąć** `franchise_frame`; werdykt o DLC/niepełnej grze klasyfikować jako `price`. (b) **dodać** `gore` — „krew, rozczłonkowanie, łamanie/miażdżenie kości, system obrażeń ciała, brutalność wizualna trafień"; granice: ogólny art direction → `graphics`, feel/impact i balans walki → `combat`. Zestaw pozostaje **15 kategorii**.

---

## D-009 — Auth w notebookach: OAuth/Bearer przez helper, koniec kluczy API

- **Status:** DONE (decyzja) / do implementacji — patrz `docs/to_do.md`
- **Zaktualizowano:** 2026-06-20 16:40 CEST
- **Kontekst:** Sonda [D-004] potwierdziła, że firmowy Vertex AI działa **wyłącznie przez OAuth**. Notebooki 03–06 wciąż używały sekretów `VERTEX_AI_API_KEY` i ścieżki `x-goog-api-key` z `_create_vertex_ai_client` (custom httpx). Dla OAuth jest to zbędne: token Bearer to dokładnie to, co domyślnie wysyła OpenAI SDK.
- **Decyzja:** W notebookach klient LLM tworzymy przez `instructor.patch(OpenAI(api_key=<token_oauth>, base_url=...))` — **bez** custom httpx i **bez** kluczy API. Token pozyskujemy jak w `00_smoke_test_auth.ipynb` (`google.colab.auth` w Colab / `google.auth.default(...)` + `refresh` lokalnie). Logika trafia do helpera `notebooks/workshop_utils.py::create_workshop_client(project_id, location)`. `src/utils/llm_client.py` zostaje w materiałach autorskich, ale **notebooki go nie importują**.
- **Następstwa:** zamyka [D-002] dla ścieżki warsztatowej (gałąź OAuth realizujemy w helperze, nie w `llm_client.py`). Konfiguracja w notebooku pozostaje widoczna: `PROJECT_ID`, `LOCATION="europe-west4"`, `MODEL="google/gemini-2.5-flash-lite"`.
- **Aktualizacja 2026-06-20 (T2 wykonane):** `create_workshop_client(project_id, location="europe-west4")` zaimplementowany w `notebooks/workshop_utils.py` — OAuth (Colab → ADC fallback), `BASE_URL` bez `/chat/completions`, zwraca `instructor.patch(OpenAI(...), mode=instructor.Mode.MD_JSON)`. Bez custom httpx, bez `x-goog-api-key`, bez kluczy API. Helper nie importuje z `src/` (zweryfikowane `grep`). Implementację dla notebooków 03–06 (podmiana starych komórek auth) realizują T3–T6.

---

## D-010 — Trym repozytorium `main` do minimum warsztatowego

- **Status:** DONE (decyzja) / do implementacji
- **Zaktualizowano:** 2026-06-20 16:40 CEST
- **Kontekst:** Uczestnicy klonują repo na starcie. Pełna struktura (scraper, scripts, analysis, docs, dane surowe, PDF-y) ich rozprasza („się pogubią"). Warsztat operuje wyłącznie na 20 recenzjach.
- **Decyzja:** Materiały autorskie przenosimy na osobną gałąź `authoring` (pełna historia bez strat). Gałąź `main` (klonowana przez uczestników) zawiera tylko: `notebooks/` (z `workshop_utils.py`), `data/evaluation/golden_dataset.{json,csv}`, odchudzony `requirements.txt`, skrócony `README`/`QUICKSTART`, `.gitignore`. `requirements.txt` redukujemy do: `openai`, `instructor`, `pydantic`, `google-auth`, `pandas`, `matplotlib`, `tqdm`.
- **Następstwa:** Wybór „jeden plik-helper" zamiast importów z `src/`. Smoke test (`00_*`) sprawdza `requirements.txt`, `notebooks`, `data` — po trymie nadal zielony.

---

## D-011 — Przebudowa dydaktyczna notebooków 03–06 (minimum informacji, puste prompty)

- **Status:** DONE (decyzja) / do implementacji
- **Zaktualizowano:** 2026-06-20 16:40 CEST
- **Kontekst:** Notebooki zdradzały rozwiązanie (gotowe prompty, słownik kategorii wstrzykiwany w prompt), miały zbędne metadane („czas trwania / poziom"), teorię wykładaną wprost (zalety/wady techniki) oraz stare 8 kategorii.
- **Decyzja:** Wspólny, odchudzony szablon dla 03–06: (1) tytuł + krótki cel bez zdradzania rozwiązania, bez metadanych czas/poziom; (2) setup OAuth; (3) jawny blok „mamy ekspercką klasyfikację 20 recenzji — odtwarzamy ją", z kolapsowalnymi tabelami kategorii+definicji oraz recenzji+etykiet (pod ręką w każdym notebooku; w 03 po setupie, w 04–06 na górze); (4) **definicja techniki bez list zalet/wad** (do dyskusji na żywo); (5) dobrze opisane zadanie bez podpowiedzi treści promptu; (6) **puste** `SYSTEM_PROMPT=""` i `USER_PROMPT=""`; (7) test na 1 recenzji; (8) klasyfikacja 20; (9) ewaluacja. Usuwamy „Anatomię promptu" z notebooka 03.
- **Następstwa / backlog (NIE do notebooków):** Świadomie pracujemy na **prostszych modelach** (`gemini-2.5-flash-lite`, bez domyślnego reasoningu, `temperature=0`), bo wymagają lepszego promptowania — to ćwiczy uczestników. Wymuszanie rozumowania (CoT) odbywa się przez pola modelu Pydantic w iteracjach 3–4, a nie przez wbudowany thinking. Ten rationale **zostaje w backlogu**, nie w treści notebooków.
- **Aktualizacja 2026-06-20 (T3 wykonane):** Szablon zrealizowany w notebooku 03 (`03_iteration1_basic_prompting.ipynb`) — wszystkie punkty decyzji spełnione: bez czas/poziom, bez „Anatomii promptu", setup OAuth, blok „mamy ekspercką klasyfikację" z kolapsowalnymi tabelami, definicja techniki bez zalet/wad, puste prompty, test→klasyfikacja 20→ewaluacja. Stanowi wzorzec dla T4–T6 (notebooki 04–06, w nich blok danych na górze).
- **Aktualizacja 2026-06-20 (T4 wykonane):** Notebook 04 (`04_iteration2_structured_output.ipynb`) przebudowany wg szablonu T3 z blokiem danych na górze. Mechanizm iteracji: `instructor` (klient już patchowany w helperze) + minimalny model Pydantic `ReviewClassification(categories: list[str])` przekazywany jako `response_model=` — `classify_review` zwraca obiekt Pydantic, nie string. Prompty puste, definicja techniki bez zalet/wad, metryki w skrócie z odwołaniem do opisu z 03. Weryfikacja: `nbconvert --to script` + `ast.parse` OK; brak `src.`; brak kluczy API.
- **Aktualizacja 2026-06-20 (T5 wykonane):** Notebook 05 (`05_iteration3_chain_of_thought.ipynb`) przebudowany wg szablonu T3 z blokiem danych na górze. Mechanizm iteracji: structured output z poprzedniej iteracji (`response_model=ReviewClassification` → obiekt Pydantic, nie string) + Chain-of-Thought wymuszany przez pole `reasoning` **przed** `categories` w modelu Pydantic. Model i prompty pozostają **pustymi placeholderami** (`class ReviewClassification(BaseModel): ...` z TODO) — uczestnik sam dodaje pole reasoning, naprowadzany treścią zadania (z wyjaśnieniem, dlaczego kolejność pól wpływa na rozumowanie), bez gotowego modelu/promptu. Metryki w skrócie z odwołaniem do opisu z 03. Weryfikacja: `nbconvert --to script` + `ast.parse` OK; brak `src.`; brak kluczy API; puste prompty obecne. Pozostają T6 (notebook 06) i T7 (trym repo).
- **Aktualizacja 2026-06-20 (T6 wykonane):** Notebook 06 (`06_iteration4_few_shot.ipynb`) przebudowany wg szablonu T3 z blokiem danych na górze. Mechanizm iteracji: few-shot — uczestnik sam dobiera przykłady w pustym placeholderze `FEW_SHOT_EXAMPLES = []` (z opisem formatu `{"review": ..., "categories": [...]}`, bez gotowych przykładów) i wplata je w puste `SYSTEM_PROMPT=""`/`USER_PROMPT=""`. Structured output z poprzednich iteracji zachowany: minimalny model `ReviewClassification(categories: list[str])` + `response_model=`. Usunięto wykładową „Teorię” (tabele zero/one/few/many-shot, few-shot vs fine-tuning) oraz końcową komórkę „konkurs 🏆” (dla spójności z jedno-modelową ewaluacją szablonu — bez `compare_trials`); README/QUICKSTART zsynchronizowane (bez „konkursu”). Metryki w skrócie z odwołaniem do opisu z 03. Weryfikacja: `nbconvert --to script` + `ast.parse` OK; brak `src.`; brak kluczy API; puste prompty + `FEW_SHOT_EXAMPLES` pusty. Notebooki 03–06 ukończone; pozostaje T7 (trym repo).

---

## D-012 — Ewaluacja: dodać czułość i swoistość (jeden model)

- **Status:** DONE (decyzja) / do implementacji
- **Zaktualizowano:** 2026-06-20 16:40 CEST
- **Kontekst:** `evaluator.py` liczy tylko accuracy (per recenzja, `CONTAINS_ALL`). To zbyt wąski obraz jakości klasyfikacji multi-label.
- **Decyzja:** Liczymy ewaluację na **jednym** modelu i raportujemy: accuracy (jak dotąd) **+ czułość (sensitivity/recall) + swoistość (specificity)** jako micro-średnią po uniwersum 15 kategorii (per recenzja: `TP=|E∩P|`, `FN=|E\P|`, `FP=|P\E|`, `TN=|U\(E∪P)|`), **oraz tabelę per-kategoria**. Dodajemy komórkę markdown tłumaczącą accuracy, czułość i swoistość. Logika w `workshop_utils.py` (port z `evaluator.py` + nowe metryki).
- **Aktualizacja 2026-06-20 (T2 wykonane):** Metryki zaimplementowane w `notebooks/workshop_utils.py`: confusion matrix multi-label vs 15 kategorii (per recenzja `TP/FN/FP/TN`), micro czułość/swoistość (`ConfusionStats`) + per-kategoria. `TrialResult.display()` pokazuje accuracy + czułość + swoistość + tabelę 15 wierszy; `compare_trials()` ma kolumny czułość/swoistość. Mini-test ręczny potwierdza zgodność z wyliczeniem (czułość 66.7%, swoistość 92.6%). Komórki markdown z opisem metryk dodają notebooki (T3 = kanoniczny opis, 04–06 = skrót).
- **Aktualizacja 2026-06-20 (T3 wykonane):** Kanoniczny opis metryk (accuracy + czułość + swoistość, micro + per-kategoria) dodany jako komórka markdown w notebooku 03; ewaluacja wołana przez `evaluate_trial(...).display()` na jednym modelu (`CONTAINS_ALL`). Notebooki 04–06 odwołają się do tego opisu w skrócie.

---

## D-013 — Finalizacja golden_dataset.{json,csv} (20 rec./15 kat.)

- **Status:** DONE (zaimplementowane)
- **Zaktualizowano:** 2026-06-20 16:40 CEST
- **Kontekst:** Etykiety w `analysis/golden_proposed.md` (20 recenzji, 15 kategorii, reguła negative-only [D-007], `gore` [D-008]) zaakceptowane przez prowadzącego. Plik `data/evaluation/golden_dataset.json` trzymał starą wersję (100 rec./8 kat.) — punkt wstrzymania z [D-006] zdjęty.
- **Decyzja:** Generujemy finalny `data/evaluation/golden_dataset.{json,csv}` skryptem `scripts/build_golden_dataset.py` (teksty z `analysis/golden_candidates.csv` po `review_id`, etykiety z `golden_proposed.md`), zastępując starą wersję. Schemat rekordu: `{id, review_id, votes_up, review_text, labels}` (klucze `review_text` i `labels` zgodne z notebookami).
- **Aktualizacja 2026-06-20 (T1 wykonane):** `scripts/build_golden_dataset.py` zaimplementowany i uruchomiony. Wygenerowano `data/evaluation/golden_dataset.{json,csv}` (20 rekordów), nadpisując starą wersję. Walidacja (20 rekordów, brak duplikatów, etykiety ⊆ 15 kat., spójność JSON↔CSV) przechodzi; rozkład pokrycia zgodny z nagłówkiem `golden_proposed.md`. Zamyka część danych z [D-006].

---

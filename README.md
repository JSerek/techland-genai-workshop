# GenAI Workshop: Klasyfikacja tekstu z LLM

**Warsztaty: Wykorzystanie GenAI w jakościowej analizie danych**

Projekt szkoleniowy do nauki klasyfikacji tekstów modelami językowymi przez API.
Dane: negatywne recenzje gry „Dying Light: The Beast” ze Steam.

---

## 📋 O co chodzi

W trakcie warsztatu odtwarzasz **ekspercką klasyfikację 20 recenzji** do **15 kategorii
tematycznych** (multi-label) — za pomocą modelu Gemini przez Vertex AI. Pracujemy w
czterech iteracjach, w każdej poprawiając prompt i metryki:

1. **Iteracja 1 — Zero-shot** (`03`): pierwsza klasyfikacja, surowy prompt.
2. **Iteracja 2 — Structured Output** (`04`): Pydantic + Instructor — odpowiedź jako walidowany obiekt z **zamkniętą listą kategorii** (`Literal`).
3. **Iteracja 3 — Chain-of-Thought** (`05`): wymuszone rozumowanie (pole `reasoning` przed kategoriami).
4. **Iteracja 4 — Few-shot** (`06`): dobór przykładów w prompcie (na bazie CoT z iteracji 3).

Każdy notebook ma **puste prompty** (multiline) do uzupełnienia — to jest właśnie ćwiczenie.
Punktem odniesienia jest golden dataset (20 recenzji z ręcznie nadanymi etykietami), a jakość
mierzymy przez **accuracy w dwóch wariantach** (`contains_all` — „zawiera wszystkie”, oraz
`is_exactly` — „idealnie”) **+ czułość + swoistość** (micro i per-kategoria), wraz z wykresem
słupkowym accuracy per kategoria.

---

## 🚀 Quickstart (Google Colab — zalecane)

Nie musisz nic instalować lokalnie. Wystarczy konto Google i dostęp do firmowego
projektu GCP z włączonym Vertex AI.

1. **Otwórz notebook** w Colab (link poda prowadzący) — zaczynamy od
   `00_smoke_test_auth.ipynb`, który sprawdza Twoje środowisko.
2. **Uruchom komórki setup.** Pierwsza klonuje repo i instaluje biblioteki
   (~1 min za pierwszym razem). Autoryzacja idzie przez **OAuth** — przy
   `authenticate_user()` zaloguj się swoim kontem Google.
3. **Wpisz `PROJECT_ID`** swojego projektu GCP w komórce konfiguracji
   (`LOCATION = "europe-west4"`, model `google/gemini-2.5-flash-lite`).
4. Przechodź notebooki **03 → 04 → 05 → 06**.

> 🔑 **Bez kluczy API.** Firmowy Vertex AI autoryzuje wyłącznie przez OAuth —
> token Bearer pozyskuje za Ciebie helper `create_workshop_client(PROJECT_ID)`.
> Token wygasa po ~1 h; przy długiej sesji wywołaj helper ponownie.

---

## 🗂️ Co jest w repo (dla uczestnika)

```
szkolenie_techland/
├── notebooks/
│   ├── 00_smoke_test_auth.ipynb          # test środowiska: OAuth + git clone
│   ├── 03_iteration1_basic_prompting.ipynb   # Iteracja 1: zero-shot
│   ├── 04_iteration2_structured_output.ipynb # Iteracja 2: Pydantic + Instructor
│   ├── 05_iteration3_chain_of_thought.ipynb  # Iteracja 3: Chain-of-Thought
│   ├── 06_iteration4_few_shot.ipynb          # Iteracja 4: few-shot
│   └── workshop_utils.py                  # helper: klient OAuth, dane, ewaluacja
├── data/
│   └── evaluation/
│       └── golden_dataset.{json,csv}      # 20 recenzji + etykiety (15 kategorii)
└── requirements.txt
```

> Helper `workshop_utils.py` leży w `notebooks/`. Notebooki dodają ten katalog do
> `sys.path` i importują `from workshop_utils import ...` — to **nie** jest import z `src/`.

---

## 🧰 Helper `workshop_utils`

Jeden plik z całą „infrastrukturą”, żeby skupić się na promptach:

| Funkcja | Do czego |
|---|---|
| `create_workshop_client(project_id, location="europe-west4")` | klient Vertex AI (OpenAI-compat) z OAuth + `instructor` |
| `load_golden()` | wczytuje golden dataset (`records`, `texts`, `labels`) |
| `show_categories()` | kolapsowalna tabela 15 kategorii z definicjami |
| `show_golden_reviews()` | kolapsowalna tabela recenzji + etykiety |
| `evaluate_trial(...)` / `compare_trials(...)` | dwa accuracy (`contains_all` + `is_exactly`) + czułość + swoistość (micro i per-kategoria) + wykres słupkowy z przełącznikiem |
| `CATEGORIES`, `MODEL`, `MatchStrategy` | słownik kategorii, nazwa modelu, strategie dopasowania |

---

## 📊 15 kategorii

`combat`, `parkour`, `enemies`, `night_horror`, `progression`, `world`, `story`,
`bugs`, `performance`, `graphics`, `audio`, `content`, `price`, `coop`, `gore`.

Pełne definicje wyświetla `show_categories()` w każdym notebooku, a osobna komórka
wypisuje kody gotowe do skopiowania (`print(", ".join(CATEGORIES))`) — np. do promptu.
**Zasada warsztatu:** w recenzji negatywnej oznaczamy tylko aspekty **krytykowane**
(pochwał nie etykietujemy).

---

## 🏗️ Lokalna instalacja (opcjonalnie)

```bash
git clone https://github.com/JSerek/techland-genai-workshop.git
cd szkolenie_techland

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Autoryzacja lokalnie przez **Application Default Credentials**:

```bash
gcloud auth application-default login
```

Następnie otwórz notebooki w `notebooks/` (cwd zwykle = `notebooks/`, więc import
helpera działa od ręki). W komórce konfiguracji wpisz swój `PROJECT_ID`.

---

## ✅ Checklist przed warsztatami

- [ ] Konto Google + dostęp do firmowego projektu GCP z Vertex AI (region `europe-west4`)
- [ ] Działa [colab.research.google.com](https://colab.research.google.com)
- [ ] `00_smoke_test_auth.ipynb` zielony (OAuth + git clone OK)
- [ ] Znasz `PROJECT_ID` swojego projektu

---

## 👤 Autor

**Jakub Serek** — szkolenie dla: Techland.
Materiały do użytku edukacyjnego.

W razie pytań podczas warsztatów — pytaj prowadzącego!

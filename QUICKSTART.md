# 🚀 Quick Start — uczestnik warsztatu

Wszystko działa w **Google Colab**. Nie instalujesz nic lokalnie.

---

## ⚡ W 4 krokach do pierwszej klasyfikacji

### 1. Otwórz `00_smoke_test_auth.ipynb`
Link poda prowadzący. Ten notebook sprawdza Twoje środowisko: logowanie OAuth do
Vertex AI oraz `git clone` repo. Uruchom wszystkie komórki — na końcu masz raport.

### 2. Uruchom komórki setup w notebooku 03
Klonują repo, instalują biblioteki (`openai`, `instructor`, `pydantic`,
`google-auth`, `pandas`, `matplotlib`, `tqdm`) i dodają `notebooks/` do `sys.path`.
Pierwsze uruchomienie ~1 min.

### 3. Zaloguj się i wpisz `PROJECT_ID`
Przy `authenticate_user()` zaloguj się kontem Google z dostępem do firmowego GCP.
W komórce konfiguracji ustaw:

```python
PROJECT_ID = "twoj-projekt-gcp"   # <- jedyne, co musisz wpisać
LOCATION   = "europe-west4"
client = create_workshop_client(PROJECT_ID, location=LOCATION)
```

> 🔑 **Bez kluczy API.** Token OAuth (Bearer) pozyskuje helper. Token wygasa po
> ~1 h — przy długiej sesji uruchom komórkę z `create_workshop_client(...)` ponownie.

### 4. Klasyfikuj
Uzupełnij `SYSTEM_PROMPT` i `USER_PROMPT`, przetestuj na 1 recenzji, puść na 20,
zobacz metryki. Potem przejdź do notebooków 04 → 05 → 06.

---

## 🗺️ Plan notebooków

| # | Notebook | Technika |
|---|----------|----------|
| 00 | `00_smoke_test_auth` | test środowiska (OAuth + git clone) |
| 03 | `03_iteration1_basic_prompting` | zero-shot |
| 04 | `04_iteration2_structured_output` | Pydantic + Instructor |
| 05 | `05_iteration3_chain_of_thought` | Chain-of-Thought |
| 06 | `06_iteration4_few_shot` | few-shot |

Każdy notebook: blok „mamy ekspercką klasyfikację 20 recenzji” (`show_categories()`,
`show_golden_reviews()`) → **puste prompty do uzupełnienia** → klasyfikacja →
ewaluacja (accuracy + czułość + swoistość).

---

## 🔧 Troubleshooting

**`from workshop_utils import ...` → ModuleNotFoundError**
Uruchom najpierw komórkę setup (dodaje `notebooks/` do `sys.path`).

**401 / 403 przy wywołaniu modelu**
Token wygasł lub brak uprawnień do projektu. Uruchom ponownie
`create_workshop_client(PROJECT_ID)` i sprawdź, czy `PROJECT_ID` jest poprawny oraz
czy masz dostęp do Vertex AI w regionie `europe-west4`.

**Pusta / dziwna odpowiedź modelu**
Najpierw testuj na 1 recenzji. Doprecyzuj prompt: format odpowiedzi i zamknięta
lista 15 kategorii (`show_categories()`).

**Golden dataset nie znaleziony**
Upewnij się, że repo zostało sklonowane (komórka setup) — plik leży w
`data/evaluation/golden_dataset.json`.

---

## ✅ Checklist przed warsztatami

- [ ] Konto Google + dostęp do firmowego projektu GCP (Vertex AI, `europe-west4`)
- [ ] Działa [colab.research.google.com](https://colab.research.google.com)
- [ ] `00_smoke_test_auth.ipynb` przechodzi na zielono
- [ ] Znasz swój `PROJECT_ID`

W razie problemów — pytaj prowadzącego! 🚀

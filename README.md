# GenAI Workshop: Text Classification with LLMs

**Warsztaty: Wykorzystanie GenAI w jakościowej analizie danych**

Projekt szkoleniowy do nauki klasyfikacji tekstów za pomocą modeli językowych poprzez API, z wykorzystaniem danych z recenzji Steam gry "Dying Light 2: The Beast".

---

## 📋 Opis Projektu

Ten projekt został stworzony na potrzeby 8-godzinnych warsztatów dotyczących wykorzystania GenAI do analizy danych jakościowych. Uczestnicy uczą się:

- **Promptowania LLM** do klasyfikacji tekstu
- **Structured Output** z wykorzystaniem Pydantic
- **Chain-of-Thought** reasoning
- **Few-shot learning** techniques
- **Ewaluacji i optymalizacji** promptów

### Program Warsztatów

1. ✅ **Wprowadzenie** - Postawy i myślenie sprzyjające wykorzystaniu GenAI
2. ✅ **Problem klasyfikacji** - Eksploracja vs. Klasyfikacja danych jakościowych
3. 🔄 **Iteracja 1**: Podstawowe promptowanie
4. 🔄 **Iteracja 2**: Structured Output (Pydantic)
5. 🔄 **Iteracja 3**: Chain-of-Thought
6. 🔄 **Iteracja 4**: Zero/One/Few-shot learning
7. 🏆 **Gamifikacja**: Konkurs na najlepszy klasyfikator

---

## 🗂️ Struktura Projektu

```
szkolenie_techland/
├── data/
│   ├── raw/                    # Surowe dane ze scrapera (gitignored)
│   ├── processed/              # Oczyszczone dane (gitignored)
│   └── evaluation/             # Zbiory ewaluacyjne
├── notebooks/
│   ├── 01_data_collection.ipynb       # ✅ Scraping + eksploracja danych
│   ├── 02_data_cleaning.ipynb         # 🔄 Przygotowanie datasetu
│   ├── 03_iteration1_basic.ipynb      # 🔄 Warsztaty: Iteracja 1
│   ├── 04_iteration2_structured.ipynb # 🔄 Warsztaty: Iteracja 2
│   ├── 05_iteration3_cot.ipynb        # 🔄 Warsztaty: Iteracja 3
│   └── 06_iteration4_few_shot.ipynb   # 🔄 Warsztaty: Iteracja 4
├── src/
│   ├── scraper/                # ✅ Moduły do scrapingu Steam
│   ├── classification/         # 🔄 Klasyfikacja z LLM
│   ├── evaluation/             # 🔄 Metryki i wizualizacje
│   └── utils/                  # ✅ Konfiguracja i narzędzia
├── scripts/
│   └── scrape_reviews.py       # ✅ CLI do szybkiego scrapingu
├── requirements.txt            # ✅ Zależności
└── README.md                   # ✅ Dokumentacja
```

---

## 🚀 Quickstart

### 👩‍💻 Uczestnicy warsztatów — Google Colab (zalecane)

Nie musisz nic instalować lokalnie. Wszystko działa w przeglądarce.

#### Krok 1: Skonfiguruj API w Colab Secrets

W każdym notebooku Colab masz menu po lewej stronie z ikoną 🔑 **Secrets**.
Dodaj tam trzy sekrety (prowadzący poda wartości na początku warsztatów):

| Nazwa sekretu | Opis |
|---|---|
| `VERTEX_AI_API_KEY` | Klucz API do modelu Gemini |
| `VERTEX_AI_BASE_URL` | Endpoint URL Vertex AI |
| `MODEL_NAME` | Nazwa modelu (np. `google/gemini-2.5-flash-lite`) |

> 💡 Sekrety są bezpieczne — nie są widoczne w notebooku ani nie trafiają do repozytorium.

#### Krok 2: Otwórz notebook

Kliknij w link do notebooka który chcesz otworzyć (linki poda prowadzący).
Alternatywnie: wejdź na [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub → wklej URL repo.

#### Krok 3: Uruchom pierwsze 3 komórki setup

Każdy notebook zaczyna się od komórek setup które:
1. Klonują repozytorium z GitHub
2. Instalują biblioteki (`openai`, `instructor`, `pydantic`, ...)
3. Wczytują API key z Secrets

> ⚠️ **Uwaga:** Pierwsze uruchomienie trwa ~1 minutę (instalacja bibliotek). Kolejne są szybkie.

---

### 🏗️ Prowadzący/Deweloperzy — lokalna instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/JSerek/techland-genai-workshop.git
cd szkolenie_techland

# Utwórz środowisko wirtualne
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Zainstaluj zależności
pip install -r requirements.txt
```

Ustaw zmienne środowiskowe (lub wpisz bezpośrednio w notebooku):
```bash
export VERTEX_AI_API_KEY="..."
export VERTEX_AI_BASE_URL="https://us-central1-aiplatform.googleapis.com/v1beta1/projects/PROJECT_ID/locations/us-central1/endpoints/openapi"
export MODEL_NAME="google/gemini-2.5-flash-lite"
```

Przygotuj dane warsztatowe (uruchom raz przed warsztatami):
```bash
# Generuje data/processed/workshop_sample.csv i szkielet golden_dataset.json
python scripts/prepare_workshop_data.py

# Przetestuj cały pipeline
python scripts/test_pipeline.py \
  --provider vertex_ai \
  --api-key "$VERTEX_AI_API_KEY" \
  --base-url "$VERTEX_AI_BASE_URL" \
  --model "$MODEL_NAME"
```

---

## 📊 Steam Review Scraper

### Funkcjonalności

- ✅ **Bez klucza API** - wykorzystuje publiczny endpoint Steam
- ✅ **Rate limiting** - bezpieczne tempo ~2 req/s
- ✅ **Checkpointing** - możliwość wznowienia po przerwaniu
- ✅ **Pydantic validation** - strukturyzowane dane
- ✅ **Multi-format export** - JSON, CSV, Parquet
- ✅ **Progress tracking** - pasek postępu z tqdm
- ✅ **Retry logic** - automatyczne ponowienie przy błędach

### Przykład Użycia (Python)

```python
from src.scraper.steam_api import SteamReviewScraper

# Inicjalizacja
scraper = SteamReviewScraper(app_id=3008130)

# Scraping
reviews = scraper.scrape_reviews(
    max_reviews=10000,
    review_type="negative",
    language="english",
    save_checkpoints=True,
)

# Statystyki
stats = scraper.get_stats_summary()
print(stats)
```

### CLI Help

```bash
python scripts/scrape_reviews.py --help
```

**Dostępne parametry:**
- `--app-id` - Steam App ID (default: 3008130)
- `--max-reviews` - Max liczba recenzji (default: 10000)
- `--review-type` - all/positive/negative (default: negative)
- `--language` - Język recenzji (default: english)
- `--formats` - Formaty eksportu (json, csv, parquet)
- `--resume` - Wznów z checkpointu
- `--no-checkpoints` - Wyłącz zapisywanie checkpointów

---

## 🎓 Dla Uczestników Warsztatów

### Przed warsztatami (zrób to dzień wcześniej):

1. **Upewnij się że masz konto Google** — potrzebne do Google Colab
2. **Wejdź na [colab.research.google.com](https://colab.research.google.com)** i sprawdź że działa
3. Nic więcej nie trzeba instalować 🎉

### Plan warsztatów:

| # | Notebook | Temat | Czas |
|---|----------|-------|------|
| 02 | `02_data_preparation` | Eksploracja danych ze Steam | 30 min |
| 03 | `03_iteration1_basic_prompting` | Zero-shot: pierwsza klasyfikacja | 35 min |
| 04 | `04_iteration2_structured_output` | Pydantic + Instructor | 45 min |
| 05 | `05_iteration3_chain_of_thought` | Chain-of-Thought reasoning | 45 min |
| 06 | `06_iteration4_few_shot` | Few-shot + konkurs 🏆 | 45 min |

Każda iteracja zawiera:
- 📚 Wprowadzenie teoretyczne (w komórkach markdown)
- 💻 Szablon kodu do uzupełnienia
- 🎯 Ćwiczenie praktyczne
- 📊 Ewaluację accuracy względem golden dataset

---

## 📦 Zależności

### Core
- `requests` - HTTP requests do Steam API
- `pydantic` - Walidacja i strukturyzacja danych
- `pandas` - Analiza danych
- `tqdm` - Progress bars

### Notebooks
- `jupyter` - Interaktywne notebooki
- `matplotlib`, `seaborn`, `plotly` - Wizualizacje

### Optional
- `pyarrow` - Wsparcie dla Parquet
- `rich` - Lepsze CLI formatowanie

---

## 🔧 Konfiguracja

Główny plik konfiguracyjny: `src/utils/config.py`

**Kluczowe ustawienia:**
```python
DYING_LIGHT_BEAST_APP_ID = 3008130
TARGET_NEGATIVE_REVIEWS = 100_000
RATE_LIMIT_DELAY = 0.5  # sekundy między requestami
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 10_000  # co ile recenzji zapisać checkpoint
```

---

## 📈 Dane

### Steam Reviews - Dying Light 2: The Beast

**Metadane zapisywane dla każdej recenzji:**
- `review_id` - Unikalny ID recenzji
- `review_text` - Treść recenzji
- `sentiment` - positive/negative
- `voted_up` - True/False
- `votes_up` - Liczba głosów "helpful"
- `playtime_hours` - Godziny gry autora
- `created_date` - Data publikacji
- `language` - Język recenzji
- `steam_purchase` - Czy zakup na Steam
- `early_access` - Czy napisane w early access

### Przykładowa recenzja:

```json
{
  "review_id": "218102167",
  "sentiment": "negative",
  "voted_up": false,
  "votes_up": 3,
  "playtime_hours": 24.8,
  "review_text": "The worst game in the series.",
  "created_date": "2026-02-11T21:50:29",
  "language": "english"
}
```

---

## 🚨 Uwagi Techniczne

### Steam API

- **Endpoint**: `https://store.steampowered.com/appreviews/{app_id}`
- **Limit**: Brak oficjalnego limitu dla publicznego endpointu
- **Rate limiting**: Zalecane max 2 req/s
- **Bez autentykacji**: Nie wymaga Steam Developer Key
- **ToS**: Używaj odpowiedzialnie, nie przeciążaj serwerów

### Checkpointing

Checkpointy są zapisywane w `data/raw/checkpoints/` co 10,000 recenzji.

Aby wznowić przerwany scraping:
```bash
python scripts/scrape_reviews.py --resume
```

### Troubleshooting

**Problem: `429 Too Many Requests`**
- Rozwiązanie: Zwiększ `RATE_LIMIT_DELAY` w config.py

**Problem: Pydantic validation errors**
- Rozwiązanie: Steam API czasem zwraca niespójne dane, retry logic powinien to obsłużyć

**Problem: Brak recenzji w określonym języku**
- Rozwiązanie: Zmień `language='all'` aby pobrać wszystkie języki

---

## 🎯 Roadmap

### Etap 1: Data Collection ✅
- [x] Steam scraper
- [x] Data models (Pydantic)
- [x] Export do multiple formats
- [x] Interactive notebook

### Etap 2: Data Preparation 🔄
- [ ] Data cleaning notebook
- [ ] Remove duplicates
- [ ] Filter low-quality reviews
- [ ] Balance dataset
- [ ] Create evaluation set

### Etap 3: Classification Setup 🔄
- [ ] Define classification categories
- [ ] LLM client wrapper
- [ ] Prompt templates
- [ ] Structured output schemas

### Etap 4: Workshops 🔄
- [ ] Iteration 1: Basic prompting
- [ ] Iteration 2: Structured outputs
- [ ] Iteration 3: Chain-of-thought
- [ ] Iteration 4: Few-shot learning

### Etap 5: Evaluation & Viz 🔄
- [ ] Custom metrics
- [ ] Confusion matrices
- [ ] Interactive dashboards
- [ ] Leaderboard system

---

## 👤 Autor

**Jakub Serek**
Szkolenie dla: Techland

## 📄 Licencja

Materiały szkoleniowe - użytek edukacyjny.

---

## 🙏 Acknowledgments

- **Steam** - za publiczne API do recenzji
- **Dying Light 2: The Beast** - dane treningowe
- **Anthropic/OpenAI** - modele LLM do klasyfikacji

---

## 📞 Kontakt / Support

W razie pytań podczas warsztatów - pytaj prowadzącego!

**Przydatne linki:**
- [Steam Web API Documentation](https://partner.steamgames.com/doc/webapi_overview)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Anthropic Claude API](https://docs.anthropic.com/)

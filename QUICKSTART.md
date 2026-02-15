# 🚀 Quick Start Guide

**Szybki start dla uczestników warsztatów**

---

## ⚡ W 3 krokach do danych

### Krok 1: Setup (5 min)

```bash
# Przejdź do folderu projektu
cd szkolenie_techland

# Utwórz środowisko wirtualne
python3 -m venv venv
source venv/bin/activate

# Zainstaluj biblioteki
pip install -r requirements.txt
```

### Krok 2: Pobierz dane (10-30 min)

**Szybki test (100 recenzji):**
```bash
python scripts/scrape_reviews.py --max-reviews 100
```

**Pełny dataset (10k recenzji - ~15 min):**
```bash
python scripts/scrape_reviews.py --max-reviews 10000
```

**Maksymalny dataset (100k recenzji - ~2h):**
```bash
python scripts/scrape_reviews.py --max-reviews 100000
```

### Krok 3: Eksploruj dane

```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

---

## 📊 Przykładowe komendy

### CLI Script - Różne opcje

```bash
# Tylko negatywne recenzje (default)
python scripts/scrape_reviews.py --max-reviews 5000

# Wszystkie recenzje (pozytywne + negatywne)
python scripts/scrape_reviews.py --max-reviews 5000 --review-type all

# Tylko pozytywne
python scripts/scrape_reviews.py --max-reviews 5000 --review-type positive

# Export tylko do JSON
python scripts/scrape_reviews.py --max-reviews 1000 --formats json

# Polskie recenzje
python scripts/scrape_reviews.py --language polish --max-reviews 1000

# Wznów przerwany scraping
python scripts/scrape_reviews.py --resume
```

---

## 🐍 Python Quick Start

```python
from src.scraper.steam_api import quick_scrape

# Najprostsze użycie
reviews = quick_scrape(
    app_id=3008130,
    max_reviews=1000,
    review_type="negative",
    language="english"
)

print(f"Pobrano {len(reviews)} recenzji")

# Zobacz pierwszą recenzję
review = reviews[0]
print(f"Sentiment: {review.sentiment}")
print(f"Text: {review.review}")
print(f"Playtime: {review.playtime_hours}h")
```

---

## 🔧 Troubleshooting

### Problem: `ModuleNotFoundError`
**Rozwiązanie:** Aktywuj środowisko wirtualne
```bash
source venv/bin/activate
```

### Problem: Scraping bardzo wolny
**Rozwiązanie:** To normalne - mamy rate limiting (bezpieczeństwo). Około 100-120 reviews/sekundę.

### Problem: `429 Too Many Requests`
**Rozwiązanie:** Steam zablokował IP na chwilę. Poczekaj 15-30 min i spróbuj ponownie.

### Problem: Scraping się przerwał
**Rozwiązanie:** Użyj `--resume` aby wznowić z ostatniego checkpointu
```bash
python scripts/scrape_reviews.py --resume
```

---

## 📁 Gdzie są dane?

Po scrapingu dane znajdziesz w:
```
data/raw/reviews_3008130_negative_english.{json,csv,parquet}
```

**Formaty:**
- `.json` - do przeglądania, debugowania
- `.csv` - do Excela, Pandas
- `.parquet` - najszybszy, najmniejszy (rekomendowany)

---

## ✅ Checklist przed warsztatami

- [ ] Zainstalowane zależności (`pip install -r requirements.txt`)
- [ ] Pobrane minimum 1000 recenzji
- [ ] Otworzony notebook `01_data_collection.ipynb`
- [ ] API Key do LLM (Claude/OpenAI/inne)
- [ ] Dostęp do Google Colab (opcjonalnie)

---

## 🆘 Pomoc

Jeśli coś nie działa:
1. Sprawdź czy środowisko wirtualne jest aktywowane
2. Sprawdź czy wszystkie zależności są zainstalowane
3. Uruchom test API: `python -c "import requests; print('OK')"`
4. Zadaj pytanie prowadzącemu!

---

## 🎯 Następne kroki

Po pobraniu danych, przejdź do:
1. `02_data_cleaning.ipynb` - czyszczenie danych
2. `03_iteration1_basic.ipynb` - pierwsze promptowanie

**Good luck!** 🚀

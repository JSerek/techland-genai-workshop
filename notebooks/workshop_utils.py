"""
Workshop Utils — helper warsztatowy (Techland / Dying Light: The Beast)
=======================================================================
Jeden samowystarczalny plik dla notebooków 03–06. **Bez importów z `src/`.**

Zawiera:
- `CATEGORIES`            — 15 kategorii (kod → definicja) + `MODEL`.
- `create_workshop_client(project_id, location="europe-west4")`
                           — klient Vertex AI (OpenAI-compat) z OAuth Bearer,
                             opatrzony `instructor` (tryb MD_JSON).
- `load_golden()`        — wczytuje golden dataset (rekordy + `texts`, `labels`).
- `show_categories()`    — kolapsowalna tabela kategorii.
- `show_golden_reviews()`— kolapsowalna tabela recenzji + etykiety.
- `MatchStrategy`, `evaluate_trial(...)`, `compare_trials(...)`
                           — ewaluacja multi-label + czułość/swoistość
                             (micro i per-kategoria, confusion matrix vs 15 kat.).

Import w notebooku (helper leży w `notebooks/`):
    # w Colab po git clone: sys.path.insert(0, f"{REPO_DIR}/notebooks")
    # lokalnie cwd zwykle = notebooks/
    from workshop_utils import (
        CATEGORIES, MODEL, create_workshop_client, load_golden,
        show_categories, show_golden_reviews,
        MatchStrategy, evaluate_trial, compare_trials,
    )
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Kategorie + model
# ---------------------------------------------------------------------------

#: 15 kategorii (kod → definicja). Źródło: analysis/aspects_with_definitions.md
CATEGORIES: dict[str, str] = {
    "combat": (
        "Walka wręcz i gunplay: feel broni, reakcje trafień/gore, balans, "
        "tryb „bestii\", boss fighty, AI w walce."
    ),
    "parkour": (
        "Traversal i ruch: płynność, wspinaczka, detekcja krawędzi, "
        "ledge-grab, momentum."
    ),
    "enemies": (
        "Bestiariusz: różnorodność i AI zombie/zarażonych, special infected, "
        "wrogowie nocni, recykling przeciwników."
    ),
    "night_horror": (
        "Cykl dzień/noc, napięcie, atmosfera, survival pressure, klimat grozy."
    ),
    "progression": (
        "Systemy RPG: skill tree, crafting, gear, loot, upgrade'y, ekonomia."
    ),
    "world": (
        "Projekt świata i eksploracja: mapa Castor Woods, gęstość/jakość side "
        "questów, fast travel, pojazd/truck, „walking simulator\"."
    ),
    "story": (
        "Fabuła, postacie (Crane), dialogi, pisanie, tempo narracji."
    ),
    "bugs": (
        "Crashe, glitche, błędy logiki, softlocki."
    ),
    "performance": (
        "FPS, stuttering, optymalizacja, ładowanie."
    ),
    "graphics": (
        "Wizualia, art direction, oświetlenie, „yellow tint\", animacje, "
        "jakość tekstur/odbić."
    ),
    "audio": (
        "Muzyka, SFX, voice acting, miks."
    ),
    "content": (
        "Ilość treści, długość kampanii, powtarzalność, replayability, endgame."
    ),
    "price": (
        "Wartość za pieniądze, polityka cenowa, DLC/edycje. Tu też werdykt „to "
        "powinno być DLC / to nie pełna gra / dodatek sprzedany za cenę pełnej gry\"."
    ),
    "coop": (
        "Tryb współdzielonej progresji, online, matchmaking, stabilność sesji."
    ),
    "gore": (
        "Efekty gore: krew, rozczłonkowanie, łamanie/miażdżenie kości, system "
        "obrażeń ciała, brutalność wizualna trafień (jakość „gore systemu\")."
    ),
}

#: Lista kodów kategorii (uniwersum dla confusion matrix).
CATEGORY_CODES: list[str] = list(CATEGORIES.keys())

#: Model (OpenAI-compatible nazwa na endpoincie Vertex AI).
MODEL = "google/gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Klient LLM (Vertex AI OpenAI-compat + OAuth Bearer + instructor)
# ---------------------------------------------------------------------------

def create_workshop_client(project_id: str, location: str = "europe-west4"):
    """Buduje klienta LLM do Vertex AI (endpoint OpenAI-compatible).

    Autoryzacja przez OAuth (Bearer) — **bez** kluczy API, **bez** custom httpx,
    **bez** nagłówka `x-goog-api-key`. Token Bearer idzie naturalnie przez OpenAI SDK.

    Token OAuth wygasa po ~1h — przy długiej sesji może być potrzebne ponowne
    wywołanie `create_workshop_client(...)`.

    Args:
        project_id: ID projektu GCP.
        location:   region Vertex AI (domyślnie ``europe-west4``).

    Returns:
        Klient OpenAI opatrzony ``instructor`` (tryb ``MD_JSON``).
    """
    if not project_id:
        raise ValueError("project_id jest wymagany (podaj ID projektu GCP).")

    # 1) Token OAuth — Colab (logowanie użytkownika) lub ADC (lokalnie / VM / SA).
    try:
        from google.colab import auth as colab_auth  # type: ignore
        colab_auth.authenticate_user()
        print("🔑 Zalogowano przez google.colab.auth")
    except ImportError:
        print("💻 Poza Colab — używam Application Default Credentials")

    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    token = creds.token

    # 2) Endpoint OpenAI-compat (BEZ /chat/completions — SDK dokleja sam).
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1beta1"
        f"/projects/{project_id}/locations/{location}/endpoints/openapi"
    )

    # 3) Klient OpenAI + instructor (MD_JSON — sprawdzony tryb na tym endpoincie).
    import instructor
    from openai import OpenAI

    client = OpenAI(api_key=token, base_url=base_url)
    return instructor.patch(client, mode=instructor.Mode.MD_JSON)


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

def _find_golden_path() -> Path:
    """Lokalizuje data/evaluation/golden_dataset.json (odpornie na Colab/lokalnie)."""
    env = os.environ.get("GOLDEN_DATASET_PATH")
    if env:
        return Path(env)

    here = Path(__file__).resolve()
    candidates = [
        # repo_root/data/... gdy helper jest w notebooks/
        here.parent.parent / "data" / "evaluation" / "golden_dataset.json",
        # cwd-względne (np. uruchomienie z repo root lub z notebooks/)
        Path.cwd() / "data" / "evaluation" / "golden_dataset.json",
        Path.cwd().parent / "data" / "evaluation" / "golden_dataset.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # zwróć pierwszy kandydat — błąd z czytelną ścieżką zgłosi load_golden()
    return candidates[0]


class GoldenData(BaseModel):
    """Golden dataset wczytany do pamięci."""
    records: list[dict]
    texts: list[str]
    labels: list[list[str]]

    def __len__(self) -> int:
        return len(self.records)


def load_golden() -> GoldenData:
    """Wczytuje golden dataset z ``data/evaluation/golden_dataset.json``.

    Returns:
        ``GoldenData`` z polami ``records`` (pełne obiekty), ``texts`` i ``labels``.
    """
    path = _find_golden_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono golden datasetu: {path}\n"
            "Ustaw zmienną środowiskową GOLDEN_DATASET_PATH lub uruchom "
            "scripts/build_golden_dataset.py."
        )

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts = [r["review_text"] for r in records]
    labels = [list(r["labels"]) for r in records]
    return GoldenData(records=records, texts=texts, labels=labels)


# ---------------------------------------------------------------------------
# Prezentacja danych „pod ręką"
# ---------------------------------------------------------------------------

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


def _details_table(summary: str, df: pd.DataFrame, open_: bool = False) -> None:
    """Renderuje kolapsowalną tabelę HTML (<details>), fallback: print."""
    if _is_notebook():
        try:
            from IPython.display import HTML, display
            open_attr = " open" if open_ else ""
            html = (
                f"<details{open_attr}>"
                f"<summary style='cursor:pointer;font-weight:bold'>{summary}</summary>"
                f"{df.to_html(index=False, escape=True)}"
                f"</details>"
            )
            display(HTML(html))
            return
        except Exception:
            pass
    print(f"\n=== {summary} ===")
    print(df.to_string(index=False))


def show_categories() -> None:
    """Wyświetla kolapsowalną tabelę 15 kategorii (kod → definicja)."""
    df = pd.DataFrame(
        [{"Kod": code, "Definicja": definicja} for code, definicja in CATEGORIES.items()]
    )
    _details_table(f"📋 Kategorie ({len(CATEGORIES)})", df)


def show_golden_reviews(max_text_len: Optional[int] = None) -> None:
    """Wyświetla kolapsowalną tabelę recenzji golden + przypisane etykiety.

    Args:
        max_text_len: jeśli podane, skraca tekst recenzji do tylu znaków.
    """
    golden = load_golden()
    rows = []
    for rec in golden.records:
        text = rec["review_text"]
        if max_text_len is not None and len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        rows.append({
            "ID": rec.get("id", ""),
            "review_id": rec.get("review_id", ""),
            "votes_up": rec.get("votes_up", ""),
            "Recenzja": text,
            "Etykiety": ", ".join(rec["labels"]),
        })
    df = pd.DataFrame(rows)
    _details_table(f"📝 Golden reviews ({len(golden)})", df)


# ---------------------------------------------------------------------------
# Ewaluacja — strategie matchowania
# ---------------------------------------------------------------------------

class MatchStrategy(str, Enum):
    """Strategie matchowania etykiet (multi-label).

    EXACT:         zbiór predykcji == zbiór oczekiwanych.
    CONTAINS_ANY:  przynajmniej jedna predykcja trafiona (recall-oriented).
    CONTAINS_ALL:  wszystkie oczekiwane są w predykcjach (dopuszcza nadmiarowe).
    PARTIAL_RATIO: Jaccard = |intersection| / |union|; próg >= 0.5 = correct.
    """
    EXACT = "exact"
    CONTAINS_ANY = "contains_any"
    CONTAINS_ALL = "contains_all"
    PARTIAL_RATIO = "partial_ratio"


def _normalize(labels: list[str]) -> set[str]:
    """Normalizuje etykiety: lowercase + strip."""
    return {label.strip().lower() for label in labels}


def _is_match(
    actual: list[str],
    expected: list[str],
    strategy: MatchStrategy,
    partial_threshold: float = 0.5,
) -> bool:
    """Sprawdza, czy predykcja jest poprawna według wybranej strategii."""
    a = _normalize(actual)
    e = _normalize(expected)

    if strategy == MatchStrategy.EXACT:
        return a == e
    elif strategy == MatchStrategy.CONTAINS_ANY:
        return bool(a & e)
    elif strategy == MatchStrategy.CONTAINS_ALL:
        return e.issubset(a)
    elif strategy == MatchStrategy.PARTIAL_RATIO:
        if not a and not e:
            return True
        union = a | e
        intersection = a & e
        return len(intersection) / len(union) >= partial_threshold
    raise ValueError(f"Nieznana strategia: {strategy}")


# ---------------------------------------------------------------------------
# Ewaluacja — modele wyników
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Wynik ewaluacji dla jednej recenzji."""
    review_text: str
    expected: list[str]
    actual: list[str]
    is_correct: bool                       # wg strategii głównej (`strategy`)
    match_strategy: MatchStrategy
    is_correct_contains_all: bool = False  # czy predykcja zawiera WSZYSTKIE oczekiwane
    is_correct_exact: bool = False         # czy predykcja == oczekiwane (idealnie)
    reasoning: Optional[str] = None        # rozumowanie modelu (CoT), jeśli zebrane


class ConfusionStats(BaseModel):
    """Confusion matrix (zliczenia + czułość/swoistość) dla agregatu lub kategorii."""
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    @property
    def sensitivity(self) -> float:
        """Czułość = TP / (TP + FN). Brak pozytywów → 0.0."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        """Swoistość = TN / (TN + FP). Brak negatywów → 0.0."""
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0


class EvaluationSummary(BaseModel):
    """Zbiorcze statystyki ewaluacji."""
    total: int
    correct: int
    accuracy: float = Field(ge=0.0, le=1.0)  # wg strategii głównej (`strategy`)
    accuracy_contains_all: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy_exact: float = Field(default=0.0, ge=0.0, le=1.0)
    results: list[EvaluationResult]
    confusion: ConfusionStats              # micro (po wszystkich kategoriach)
    per_category: dict[str, ConfusionStats]  # kod kategorii → confusion


class TrialResult(BaseModel):
    """Wyniki jednej próby (kombinacja prompt + model)."""
    trial_name: str = Field(description="Czytelna nazwa próby, np. 'gemini + zero-shot'")
    model: str
    prompt_variant: str
    summary: EvaluationSummary

    def display(self, max_text_len: int = 80) -> None:
        """Tabelka wyników + accuracy + czułość/swoistość (micro) + per-kategoria."""
        _display_trial_table(self, max_text_len=max_text_len)

    def plot(self) -> None:
        """Wykres accuracy: contains_all vs is_exactly (osobna komórka)."""
        _plot_accuracy_overall(self.summary)


# ---------------------------------------------------------------------------
# Ewaluacja — funkcje główne
# ---------------------------------------------------------------------------

def _confusion_for_review(
    actual: set[str], expected: set[str], universe: set[str]
) -> dict[str, str]:
    """Mapuje każdą kategorię z uniwersum na 'tp'/'fn'/'fp'/'tn' dla jednej recenzji."""
    out = {}
    for cat in universe:
        in_e = cat in expected
        in_p = cat in actual
        if in_e and in_p:
            out[cat] = "tp"
        elif in_e and not in_p:
            out[cat] = "fn"
        elif not in_e and in_p:
            out[cat] = "fp"
        else:
            out[cat] = "tn"
    return out


def evaluate_trial(
    trial_name: str,
    model: str,
    prompt_variant: str,
    predictions: list[list[str]],
    expected: list[list[str]],
    review_texts: list[str],
    strategy: MatchStrategy = MatchStrategy.CONTAINS_ALL,
    categories: Optional[list[str]] = None,
    reasonings: Optional[list[str]] = None,
) -> TrialResult:
    """Ewaluuje wyniki jednej próby klasyfikacji multi-label.

    Liczy:
    - accuracy (wg ``strategy``) oraz dwa warianty niezależnie od strategii:
      ``accuracy_contains_all`` (predykcja zawiera wszystkie oczekiwane) i
      ``accuracy_exact`` (predykcja dokładnie równa oczekiwanym),
    - confusion matrix względem uniwersum kategorii (per recenzja:
      ``TP=|E∩P|, FN=|E\\P|, FP=|P\\E|, TN=|U\\(E∪P)|``),
    - micro czułość ``ΣTP/(ΣTP+ΣFN)`` i swoistość ``ΣTN/(ΣTN+ΣFP)``,
    - czułość/swoistość per-kategoria.

    Args:
        categories: uniwersum kategorii (domyślnie 15 kodów z ``CATEGORIES``).
        reasonings: opcjonalne rozumowania modelu (CoT) — po jednym na recenzję;
            jeśli podane, trafiają do tabeli wyników.
    """
    if not (len(predictions) == len(expected) == len(review_texts)):
        raise ValueError(
            f"Długości list muszą być równe: predictions={len(predictions)}, "
            f"expected={len(expected)}, review_texts={len(review_texts)}"
        )
    if reasonings is not None and len(reasonings) != len(predictions):
        raise ValueError(
            f"reasonings musi mieć tyle samo elementów co predictions "
            f"({len(reasonings)} != {len(predictions)})."
        )

    universe = {c.strip().lower() for c in (categories or CATEGORY_CODES)}

    results: list[EvaluationResult] = []
    per_cat: dict[str, ConfusionStats] = {c: ConfusionStats() for c in universe}
    micro = ConfusionStats()

    for i, (text, pred, exp) in enumerate(zip(review_texts, predictions, expected)):
        a = _normalize(pred)
        e = _normalize(exp)
        correct = _is_match(pred, exp, strategy)
        results.append(EvaluationResult(
            review_text=text,
            expected=exp,
            actual=pred,
            is_correct=correct,
            match_strategy=strategy,
            is_correct_contains_all=_is_match(pred, exp, MatchStrategy.CONTAINS_ALL),
            is_correct_exact=_is_match(pred, exp, MatchStrategy.EXACT),
            reasoning=(reasonings[i] if reasonings is not None else None),
        ))

        cell = _confusion_for_review(a, e, universe)
        for cat, kind in cell.items():
            setattr(per_cat[cat], kind, getattr(per_cat[cat], kind) + 1)
            setattr(micro, kind, getattr(micro, kind) + 1)

    correct_count = sum(r.is_correct for r in results)
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    acc_contains_all = (
        sum(r.is_correct_contains_all for r in results) / total if total else 0.0
    )
    acc_exact = sum(r.is_correct_exact for r in results) / total if total else 0.0

    summary = EvaluationSummary(
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 4),
        accuracy_contains_all=round(acc_contains_all, 4),
        accuracy_exact=round(acc_exact, 4),
        results=results,
        confusion=micro,
        per_category=per_cat,
    )
    return TrialResult(
        trial_name=trial_name,
        model=model,
        prompt_variant=prompt_variant,
        summary=summary,
    )


def compare_trials(trials: list[TrialResult]) -> None:
    """Porównuje do 3 prób: tabelka zbiorcza + wykres słupkowy accuracy."""
    if not trials:
        print("Brak prób do porównania.")
        return
    if len(trials) > 3:
        print("⚠️  Uwaga: podano więcej niż 3 próby. Pokazuję pierwsze 3.")
        trials = trials[:3]

    _display_comparison_table(trials)
    _display_comparison_chart(trials)


# ---------------------------------------------------------------------------
# Ewaluacja — display helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _reasoning_details(text: str) -> str:
    """Reasoning (CoT) jako zwijany <details>: skrót w nagłówku, pełny tekst po kliknięciu."""
    import html as _html
    text = (text or "").strip()
    if not text:
        return ""
    head = text[:80] + ("…" if len(text) > 80 else "")
    return (
        f"<details><summary style='cursor:pointer'>{_html.escape(head)}</summary>"
        f"<div style='white-space:pre-wrap;text-align:left;max-width:560px'>"
        f"{_html.escape(text)}</div></details>"
    )


def _per_category_df(summary: EvaluationSummary) -> pd.DataFrame:
    """Tabela per-kategoria: TP/FN/FP/TN + czułość + swoistość (wiersz na kategorię)."""
    rows = []
    for cat in CATEGORY_CODES:
        cs = summary.per_category.get(cat, ConfusionStats())
        rows.append({
            "Kategoria": cat,
            "TP": cs.tp, "FN": cs.fn, "FP": cs.fp, "TN": cs.tn,
            "Czułość": f"{cs.sensitivity:.1%}",
            "Swoistość": f"{cs.specificity:.1%}",
        })
    # uwzględnij ewentualne kategorie spoza CATEGORY_CODES (nie powinno wystąpić)
    for cat in summary.per_category:
        if cat not in CATEGORY_CODES:
            cs = summary.per_category[cat]
            rows.append({
                "Kategoria": cat,
                "TP": cs.tp, "FN": cs.fn, "FP": cs.fp, "TN": cs.tn,
                "Czułość": f"{cs.sensitivity:.1%}",
                "Swoistość": f"{cs.specificity:.1%}",
            })
    return pd.DataFrame(rows)


def _display_trial_table(trial: TrialResult, max_text_len: int = 80) -> None:
    """Wyświetla wyniki jednej próby: per recenzja + metryki + per-kategoria (bez wykresu)."""
    summary = trial.summary
    micro = summary.confusion
    total = summary.total

    n_ca = sum(r.is_correct_contains_all for r in summary.results)
    n_ex = sum(r.is_correct_exact for r in summary.results)

    print(f"\n{'='*70}")
    print(f"Próba: {trial.trial_name}")
    print(f"Model: {trial.model} | Prompt: {trial.prompt_variant}")
    print(f"{'='*70}")
    print(f"Accuracy (zawiera wszystkie / contains_all): "
          f"{summary.accuracy_contains_all:.1%}  ({n_ca}/{total})")
    print(f"Accuracy (idealnie / is_exactly):            "
          f"{summary.accuracy_exact:.1%}  ({n_ex}/{total})")
    print(f"Czułość (micro):   {micro.sensitivity:.1%}  "
          f"(TP={micro.tp}, FN={micro.fn})")
    print(f"Swoistość (micro): {micro.specificity:.1%}  "
          f"(TN={micro.tn}, FP={micro.fp})")
    print(f"{'='*70}\n")

    # Czy mamy rozumowanie (CoT) do pokazania?
    has_reasoning = any((r.reasoning or "").strip() for r in summary.results)

    # Tabela per-recenzja
    rows = []
    for r in summary.results:
        row = {
            "Recenzja (fragment)": _truncate(r.review_text, max_text_len),
            "Oczekiwane": ", ".join(r.expected),
            "Predykcja modelu": ", ".join(r.actual),
            "Poprawne (zawiera wszystkie)": "✅" if r.is_correct_contains_all else "❌",
            "Poprawne (idealnie)": "✅" if r.is_correct_exact else "❌",
        }
        if has_reasoning:
            row["Rozumowanie (CoT)"] = r.reasoning or ""  # pełny tekst — zwijany w tabeli
        rows.append(row)
    df = pd.DataFrame(rows)

    per_cat_df = _per_category_df(summary)
    correct_cols = ["Poprawne (zawiera wszystkie)", "Poprawne (idealnie)"]

    try:
        from IPython.display import display
        styled = df.style.apply(
            lambda col: [
                "background-color: #d4edda" if v == "✅" else "background-color: #f8d7da"
                for v in col
            ],
            subset=correct_cols,
        )
        if has_reasoning:
            # zwijane <details> z pełnym reasoningiem (Styler domyślnie nie escapuje HTML)
            styled = styled.format({"Rozumowanie (CoT)": _reasoning_details})
        display(styled)
        print("\nMetryki per-kategoria:")
        display(per_cat_df.style.hide(axis="index"))
    except ImportError:
        with pd.option_context("display.max_colwidth", None):
            print(df.to_string(index=False))
        print("\nMetryki per-kategoria:")
        print(per_cat_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Ewaluacja — wykres accuracy (contains_all vs is_exactly)
# ---------------------------------------------------------------------------

def _plot_accuracy_overall(summary: EvaluationSummary) -> None:
    """Jeden wykres słupkowy: accuracy całej recenzji — contains_all vs is_exactly."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nie jest zainstalowany - pomijam wykres.")
        return

    labels = ["Zawiera wszystkie\n(contains_all)", "Idealnie\n(is_exactly)"]
    values = [summary.accuracy_contains_all * 100.0, summary.accuracy_exact * 100.0]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy: contains_all vs is_exactly",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def _display_comparison_table(trials: list[TrialResult]) -> None:
    """Zbiorcza tabelka: accuracy + czułość + swoistość (micro) wszystkich prób."""
    print(f"\n{'='*50}")
    print("PORÓWNANIE PRÓB")
    print(f"{'='*50}")

    rows = []
    for t in trials:
        m = t.summary.confusion
        rows.append({
            "Próba": t.trial_name,
            "Model": t.model,
            "Prompt": t.prompt_variant,
            "Accuracy": f"{t.summary.accuracy:.1%}",
            "Czułość": f"{m.sensitivity:.1%}",
            "Swoistość": f"{m.specificity:.1%}",
        })
    df = pd.DataFrame(rows)

    try:
        from IPython.display import display
        display(df.style.hide(axis="index"))
    except ImportError:
        print(df.to_string(index=False))


def _display_comparison_chart(trials: list[TrialResult]) -> None:
    """Wykres słupkowy porównujący accuracy prób."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nie jest zainstalowany - pomijam wykres.")
        return

    names = [t.trial_name for t in trials]
    accuracies = [t.summary.accuracy * 100 for t in trials]
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(trials)]

    fig, ax = plt.subplots(figsize=(max(6, len(trials) * 2.5), 5))
    bars = ax.bar(names, accuracies, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Porównanie accuracy prób", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, wrap=True)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

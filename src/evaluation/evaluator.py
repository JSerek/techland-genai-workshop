"""
Evaluation Framework
====================
Moduł do ewaluacji wyników klasyfikacji LLM względem golden dataset.

Obsługuje:
- Multi-label klasyfikację (jedna recenzja → wiele kategorii)
- 4 strategie matchowania stringów
- Porównanie do 3 prób (prompt/model combinations) jednocześnie
- Tabelkę wyników per próba
- Wykres słupkowy porównujący accuracy prób

Użycie:
    from src.evaluation.evaluator import evaluate_trial, compare_trials, MatchStrategy

    trial = evaluate_trial(
        trial_name="gpt-4o-mini + zero-shot",
        model="gpt-4o-mini",
        prompt_variant="zero-shot v1",
        predictions=[["bug", "performance"], ["story"]],
        expected=[["bug"], ["story", "gameplay"]],
        review_texts=["The game crashes...", "Love the story..."],
        strategy=MatchStrategy.CONTAINS_ALL,
    )
    trial.display()

    # Porównanie wielu prób:
    compare_trials([trial1, trial2, trial3])
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Match Strategies
# ---------------------------------------------------------------------------

class MatchStrategy(str, Enum):
    """
    Strategie matchowania etykiet (multi-label).

    EXACT:        Zbiór predykcji == zbiór oczekiwanych (wszystkie muszą się zgadzać, żadna nadmiarowa).
    CONTAINS_ANY: Przynajmniej jedna predykcja trafiona (recall-oriented).
    CONTAINS_ALL: Wszystkie oczekiwane etykiety są w predykcjach (dopuszcza nadmiarowe).
    PARTIAL_RATIO: Wynik Jaccard = |intersection| / |union|; próg >= 0.5 = correct.
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
        return bool(a & e)  # intersection nie jest pusta

    elif strategy == MatchStrategy.CONTAINS_ALL:
        return e.issubset(a)  # wszystkie oczekiwane są w predykcjach

    elif strategy == MatchStrategy.PARTIAL_RATIO:
        if not a and not e:
            return True
        union = a | e
        intersection = a & e
        jaccard = len(intersection) / len(union)
        return jaccard >= partial_threshold

    raise ValueError(f"Nieznana strategia: {strategy}")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Wynik ewaluacji dla jednej recenzji."""
    review_text: str
    expected: list[str]
    actual: list[str]
    is_correct: bool
    match_strategy: MatchStrategy


class EvaluationSummary(BaseModel):
    """Zbiorcze statystyki ewaluacji."""
    total: int
    correct: int
    accuracy: float = Field(ge=0.0, le=1.0)
    results: list[EvaluationResult]


class TrialResult(BaseModel):
    """Wyniki jednej próby (kombinacja prompt + model)."""
    trial_name: str = Field(description="Czytelna nazwa próby, np. 'gpt-4o-mini + zero-shot'")
    model: str
    prompt_variant: str
    summary: EvaluationSummary

    def display(self, max_text_len: int = 80) -> None:
        """Wyświetla tabelkę wyników i statystyki dla tej próby."""
        _display_trial_table(self, max_text_len=max_text_len)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def evaluate_trial(
    trial_name: str,
    model: str,
    prompt_variant: str,
    predictions: list[list[str]],
    expected: list[list[str]],
    review_texts: list[str],
    strategy: MatchStrategy = MatchStrategy.CONTAINS_ALL,
) -> TrialResult:
    """
    Ewaluuje wyniki jednej próby klasyfikacji.

    Args:
        trial_name:    Czytelna nazwa próby.
        model:         Nazwa modelu, np. "gpt-4o-mini".
        prompt_variant: Opis wariantu promptu, np. "zero-shot v1".
        predictions:   Lista list predykowanych etykiet (jedna lista per recenzja).
        expected:      Lista list oczekiwanych etykiet z golden dataset.
        review_texts:  Oryginalne teksty recenzji.
        strategy:      Strategia matchowania etykiet.

    Returns:
        TrialResult z pełnymi wynikami i statystykami.

    Przykład:
        trial = evaluate_trial(
            trial_name="gemini-flash + basic prompt",
            model="gemini-1.5-flash",
            prompt_variant="zero-shot",
            predictions=[["bug", "performance"], ["story"]],
            expected=[["bug"], ["story", "gameplay"]],
            review_texts=["The game crashes...", "Great narrative..."],
        )
    """
    if not (len(predictions) == len(expected) == len(review_texts)):
        raise ValueError(
            f"Długości lists muszą być równe: "
            f"predictions={len(predictions)}, expected={len(expected)}, "
            f"review_texts={len(review_texts)}"
        )

    results = []
    for text, pred, exp in zip(review_texts, predictions, expected):
        correct = _is_match(pred, exp, strategy)
        results.append(EvaluationResult(
            review_text=text,
            expected=exp,
            actual=pred,
            is_correct=correct,
            match_strategy=strategy,
        ))

    correct_count = sum(r.is_correct for r in results)
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    summary = EvaluationSummary(
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 4),
        results=results,
    )

    return TrialResult(
        trial_name=trial_name,
        model=model,
        prompt_variant=prompt_variant,
        summary=summary,
    )


def compare_trials(trials: list[TrialResult]) -> None:
    """
    Porównuje do 3 prób: tabelka zbiorcza + wykres słupkowy.

    Args:
        trials: Lista TrialResult (max 3).
    """
    if not trials:
        print("Brak prób do porównania.")
        return
    if len(trials) > 3:
        print("⚠️  Uwaga: podano więcej niż 3 próby. Pokazuję pierwsze 3.")
        trials = trials[:3]

    _display_comparison_table(trials)
    _display_comparison_chart(trials)


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _display_trial_table(trial: TrialResult, max_text_len: int = 80) -> None:
    """Wyświetla tabelkę wyników dla jednej próby."""
    summary = trial.summary

    print(f"\n{'='*70}")
    print(f"Próba: {trial.trial_name}")
    print(f"Model: {trial.model} | Prompt: {trial.prompt_variant}")
    print(f"Strategia matchowania: {summary.results[0].match_strategy.value if summary.results else 'N/A'}")
    print(f"{'='*70}")
    print(f"Wynik: {summary.correct}/{summary.total} poprawnych  |  Accuracy: {summary.accuracy:.1%}")
    print(f"{'='*70}\n")

    rows = []
    for r in summary.results:
        rows.append({
            "Recenzja (fragment)": _truncate(r.review_text, max_text_len),
            "Oczekiwane": ", ".join(r.expected),
            "Predykcja modelu": ", ".join(r.actual),
            "Poprawne": "✅" if r.is_correct else "❌",
        })

    df = pd.DataFrame(rows)

    try:
        # Kolorowanie w Jupyter/Colab
        from IPython.display import display
        styled = df.style.apply(
            lambda col: [
                "background-color: #d4edda" if v == "✅" else "background-color: #f8d7da"
                for v in col
            ],
            subset=["Poprawne"]
        )
        display(styled)
    except ImportError:
        print(df.to_string(index=False))


def _display_comparison_table(trials: list[TrialResult]) -> None:
    """Wyświetla zbiorczą tabelkę accuracy wszystkich prób."""
    print(f"\n{'='*50}")
    print("PORÓWNANIE PRÓB")
    print(f"{'='*50}")

    rows = []
    for t in trials:
        rows.append({
            "Próba": t.trial_name,
            "Model": t.model,
            "Prompt": t.prompt_variant,
            "Correct": f"{t.summary.correct}/{t.summary.total}",
            "Accuracy": f"{t.summary.accuracy:.1%}",
        })

    df = pd.DataFrame(rows)

    try:
        from IPython.display import display
        display(df.style.hide(axis="index"))
    except ImportError:
        print(df.to_string(index=False))


def _display_comparison_chart(trials: list[TrialResult]) -> None:
    """Wyświetla wykres słupkowy porównujący accuracy prób."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib nie jest zainstalowany - pomijam wykres.")
        return

    names = [t.trial_name for t in trials]
    accuracies = [t.summary.accuracy * 100 for t in trials]  # w procentach

    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(trials)]

    fig, ax = plt.subplots(figsize=(max(6, len(trials) * 2.5), 5))

    bars = ax.bar(names, accuracies, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    # Etykiety wartości na słupkach
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

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

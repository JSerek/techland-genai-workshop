"""
End-to-end test całego pipeline'u warsztatowego.

Testuje:
1. Import modułów (llm_client, evaluator)
2. Połączenie z API LLM
3. Klasyfikację przykładowej recenzji (structured output)
4. Ewaluację na golden dataset
5. Porównanie 2 wariantów promptów (compare_trials)

Użycie:
    # Vertex AI (Gemini):
    python scripts/test_pipeline.py \
        --provider vertex_ai \
        --api-key "ya29..." \
        --base-url "https://us-central1-aiplatform.googleapis.com/v1/projects/MY_PROJECT/locations/us-central1/endpoints/openapi" \
        --model "gemini-1.5-flash"

    # OpenAI:
    python scripts/test_pipeline.py \
        --provider openai \
        --api-key "sk-..." \
        --model "gpt-4o-mini"
"""

import argparse
import json
import sys
from pathlib import Path

# Dodaj root do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    print("\n[1/5] Test importów...")
    from src.utils.llm_client import create_client, LLMProvider
    from src.evaluation.evaluator import (
        evaluate_trial, compare_trials, MatchStrategy,
        EvaluationResult, TrialResult
    )
    from pydantic import BaseModel, Field
    print("  ✅ Wszystkie importy OK")
    return True


def test_api_connection(provider_str, api_key, base_url, model):
    print("\n[2/5] Test połączenia z API...")
    from src.utils.llm_client import create_client, LLMProvider

    provider = LLMProvider(provider_str)
    client = create_client(provider=provider, api_key=api_key, base_url=base_url)

    # Prosty ping - użyj create_client (ma już właściwe nagłówki per provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with just: OK"}],
        max_tokens=5,
    )
    reply = response.choices[0].message.content.strip()
    print(f"  ✅ API odpowiedział: '{reply}'")
    return client


def test_structured_output(client, model):
    print("\n[3/5] Test Structured Output (Pydantic + Instructor)...")
    from pydantic import BaseModel, Field

    class TestClassification(BaseModel):
        categories: list[str] = Field(description="Lista kategorii z recenzji")
        reasoning: str = Field(description="Krótkie uzasadnienie")

    test_review = (
        "The game crashes every 20 minutes and the FPS drops to 10 "
        "in crowded areas. Unplayable on my high-end PC."
    )

    result = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify this game review into categories. "
                    "Available: bug, performance, story, gameplay, graphics, audio, price, other"
                )
            },
            {"role": "user", "content": f"Review: {test_review}"},
        ],
        response_model=TestClassification,
        temperature=0.0,
    )

    print(f"  ✅ Structured output działa!")
    print(f"     Kategorie: {result.categories}")
    print(f"     Reasoning: {result.reasoning[:80]}...")
    return result


def test_evaluator():
    print("\n[4/5] Test modułu ewaluacji...")
    from src.evaluation.evaluator import evaluate_trial, MatchStrategy

    # Mock dane - bez API
    mock_reviews = [
        "Game crashes constantly, terrible performance",
        "Beautiful graphics but shallow story",
        "Best gameplay mechanics in years, smooth 60fps",
    ]
    mock_predictions = [
        ["bug", "performance"],
        ["graphics"],               # brakuje "story" z expected → błąd
        ["gameplay"],
    ]
    mock_expected = [
        ["bug", "performance"],
        ["graphics", "story"],      # model nie przewidział "story" → incorrect
        ["gameplay"],
    ]

    trial = evaluate_trial(
        trial_name="Test Trial",
        model="test-model",
        prompt_variant="test",
        predictions=mock_predictions,
        expected=mock_expected,
        review_texts=mock_reviews,
        strategy=MatchStrategy.CONTAINS_ALL,  # expected ⊆ actual
    )

    assert trial.summary.total == 3
    assert trial.summary.correct == 2  # recenzja 2 błędna bo brakuje "story"
    assert abs(trial.summary.accuracy - 2/3) < 0.01

    print(f"  ✅ Ewaluator działa!")
    print(f"     Accuracy: {trial.summary.accuracy:.1%} ({trial.summary.correct}/{trial.summary.total})")
    return trial


def test_full_evaluation(client, model, golden_dataset_path: str):
    print("\n[5/5] Test pełnej ewaluacji na golden dataset...")

    golden_path = Path(golden_dataset_path)
    if not golden_path.exists():
        print(f"  ⚠️  Golden dataset nie znaleziony: {golden_dataset_path}")
        print("     Uruchom najpierw: python scripts/prepare_workshop_data.py --mock-labels --categories bug performance story gameplay graphics")
        return None

    with open(golden_path) as f:
        golden_data = json.load(f)

    # Filtruj tylko przykłady z etykietami
    labeled = [d for d in golden_data if d.get("labels")]
    if not labeled:
        print("  ⚠️  Brak etykiet w golden dataset. Uzupełnij pole 'labels'.")
        return None

    from src.evaluation.evaluator import evaluate_trial, compare_trials, MatchStrategy
    from pydantic import BaseModel, Field

    class Classification(BaseModel):
        reasoning: str = Field(description="Step-by-step analysis")
        categories: list[str] = Field(description="List of topic categories")

    CATEGORIES = list({label for item in labeled for label in item["labels"]})
    categories_str = ", ".join(f'"{c}"' for c in CATEGORIES)

    # Wariant 1: Prosty prompt
    print(f"  Klasyfikuję {len(labeled)} recenzji (wariant 1: prosty prompt)...")
    preds_simple = []
    for item in labeled:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Classify game review. Categories: {categories_str}"},
                    {"role": "user", "content": item["review_text"]},
                ],
                response_model=Classification,
                temperature=0.0,
            )
            preds_simple.append(r.categories)
        except Exception as e:
            print(f"    Błąd: {e}")
            preds_simple.append([])

    # Wariant 2: CoT prompt
    print(f"  Klasyfikuję {len(labeled)} recenzji (wariant 2: CoT prompt)...")
    preds_cot = []
    for item in labeled:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a game review analyst. Classify reviews into topic categories.\n"
                            f"Categories: {categories_str}\n\n"
                            f"Think step by step: 1) What does the review describe? "
                            f"2) Which categories match? 3) Final answer."
                        )
                    },
                    {"role": "user", "content": f"Review: {item['review_text']}"},
                ],
                response_model=Classification,
                temperature=0.0,
            )
            preds_cot.append(r.categories)
        except Exception as e:
            print(f"    Błąd: {e}")
            preds_cot.append([])

    golden_texts = [item["review_text"] for item in labeled]
    golden_labels = [item["labels"] for item in labeled]

    trial1 = evaluate_trial(
        trial_name=f"{model} + simple prompt",
        model=model, prompt_variant="simple",
        predictions=preds_simple, expected=golden_labels,
        review_texts=golden_texts, strategy=MatchStrategy.CONTAINS_ALL,
    )
    trial2 = evaluate_trial(
        trial_name=f"{model} + CoT prompt",
        model=model, prompt_variant="CoT",
        predictions=preds_cot, expected=golden_labels,
        review_texts=golden_texts, strategy=MatchStrategy.CONTAINS_ALL,
    )

    print(f"\n  ✅ Ewaluacja zakończona!")
    compare_trials([trial1, trial2])
    return trial1, trial2


def main():
    parser = argparse.ArgumentParser(description="End-to-end test pipeline'u warsztatowego")
    parser.add_argument("--provider", default="vertex_ai",
                        choices=["openai", "anthropic", "vertex_ai", "azure_openai"])
    parser.add_argument("--api-key", required=True, help="API key / access token")
    parser.add_argument("--base-url", default=None, help="Endpoint URL (Vertex AI / Azure)")
    parser.add_argument("--model", default="gemini-1.5-flash")
    parser.add_argument("--golden-dataset", default="data/evaluation/golden_dataset.json")
    parser.add_argument("--skip-full-eval", action="store_true",
                        help="Pomiń pełną ewaluację (tylko test importów + API + structured output)")
    args = parser.parse_args()

    print("=" * 60)
    print("END-TO-END TEST PIPELINE'U WARSZTATOWEGO")
    print(f"Provider: {args.provider} | Model: {args.model}")
    print("=" * 60)

    results = {}

    try:
        results["imports"] = test_imports()
    except Exception as e:
        print(f"  ❌ BŁĄD: {e}")
        sys.exit(1)

    try:
        client = test_api_connection(args.provider, args.api_key, args.base_url, args.model)
        results["api"] = True
    except Exception as e:
        print(f"  ❌ BŁĄD połączenia z API: {e}")
        sys.exit(1)

    try:
        test_structured_output(client, args.model)
        results["structured"] = True
    except Exception as e:
        print(f"  ❌ BŁĄD structured output: {e}")
        sys.exit(1)

    try:
        test_evaluator()
        results["evaluator"] = True
    except AssertionError as e:
        print(f"  ❌ BŁĄD ewaluatora (assertion): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ BŁĄD ewaluatora: {e}")
        sys.exit(1)

    if not args.skip_full_eval:
        try:
            test_full_evaluation(client, args.model, args.golden_dataset)
            results["full_eval"] = True
        except Exception as e:
            print(f"  ❌ BŁĄD pełnej ewaluacji: {e}")

    print("\n" + "=" * 60)
    print("PODSUMOWANIE TESTÓW:")
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

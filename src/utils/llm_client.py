"""
LLM Client Router
=================
Jednolity interfejs do różnych providerów LLM: OpenAI, Anthropic, Vertex AI (Gemini), Azure OpenAI.
Zwraca klienta opatrzonego przez Instructor - gotowego do Structured Output z Pydantic.

Użycie:
    from src.utils.llm_client import create_client, LLMProvider

    client = create_client(
        provider=LLMProvider.VERTEX_AI,
        api_key="...",
        base_url="https://...",
    )
"""

from enum import Enum
from typing import Optional

import instructor


class LLMProvider(str, Enum):
    """Wspierani providerzy LLM."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"       # Gemini via Vertex AI (OpenAI-compatible endpoint)
    AZURE_OPENAI = "azure_openai"


def create_client(
    provider: LLMProvider,
    api_key: str,
    base_url: Optional[str] = None,
    instructor_mode: Optional[instructor.Mode] = None,
) -> instructor.Instructor:
    """
    Tworzy klienta LLM z Instructor dla danego providera.

    Args:
        provider:        Wybrany provider (LLMProvider enum).
        api_key:         Klucz API lub token dostępu.
        base_url:        Endpoint URL (wymagany dla Vertex AI i Azure OpenAI).
        instructor_mode: Tryb Instructor (domyślnie dobierany per provider).

    Returns:
        Instructor-patched client gotowy do structured output.

    Przykłady:
        # OpenAI
        client = create_client(LLMProvider.OPENAI, api_key="sk-...")

        # Vertex AI (Gemini) - OpenAI-compatible endpoint
        client = create_client(
            LLMProvider.VERTEX_AI,
            api_key="ya29...",  # Google access token
            base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1/endpoints/openapi",
        )

        # Anthropic
        client = create_client(LLMProvider.ANTHROPIC, api_key="sk-ant-...")

        # Azure OpenAI
        client = create_client(
            LLMProvider.AZURE_OPENAI,
            api_key="...",
            base_url="https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT",
        )
    """
    if provider == LLMProvider.OPENAI:
        return _create_openai_client(api_key=api_key, base_url=base_url, mode=instructor_mode)

    elif provider == LLMProvider.VERTEX_AI:
        if not base_url:
            raise ValueError(
                "base_url jest wymagany dla Vertex AI.\n"
                "Format: https://REGION-aiplatform.googleapis.com/v1beta1/projects/PROJECT_ID/"
                "locations/REGION/endpoints/openapi"
            )
        # Vertex AI używa x-goog-api-key zamiast Authorization: Bearer.
        # OpenAI SDK zawsze wysyła Authorization: Bearer, dlatego używamy
        # custom httpx client który podmienia nagłówki autoryzacji.
        return _create_vertex_ai_client(api_key=api_key, base_url=base_url, mode=instructor_mode)

    elif provider == LLMProvider.AZURE_OPENAI:
        if not base_url:
            raise ValueError(
                "base_url jest wymagany dla Azure OpenAI.\n"
                "Format: https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT"
            )
        return _create_openai_client(api_key=api_key, base_url=base_url, mode=instructor_mode)

    elif provider == LLMProvider.ANTHROPIC:
        return _create_anthropic_client(api_key=api_key, mode=instructor_mode)

    else:
        raise ValueError(f"Nieznany provider: {provider}. Dostępne: {list(LLMProvider)}")


def _create_openai_client(
    api_key: str,
    base_url: Optional[str],
    mode: Optional[instructor.Mode],
) -> instructor.Instructor:
    """Tworzy OpenAI-compatible client (OpenAI, Azure)."""
    from openai import OpenAI

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    raw_client = OpenAI(**kwargs)
    selected_mode = mode or instructor.Mode.MD_JSON
    return instructor.patch(raw_client, mode=selected_mode)


def _create_vertex_ai_client(
    api_key: str,
    base_url: str,
    mode: Optional[instructor.Mode],
) -> instructor.Instructor:
    """
    Tworzy Vertex AI client z prawidłową autoryzacją.

    Vertex AI wymaga nagłówka x-goog-api-key zamiast Authorization: Bearer.
    OpenAI SDK zawsze dodaje Bearer, dlatego używamy custom httpx klienta
    który podmienia nagłówki przed wysłaniem requestu.
    """
    import httpx
    from openai import OpenAI

    class _VertexAIHttpClient(httpx.Client):
        """Custom httpx client który zamienia Bearer auth na x-goog-api-key."""

        def __init__(self, vertex_api_key: str, **kwargs):
            super().__init__(**kwargs)
            self._vertex_api_key = vertex_api_key

        def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
            # Usuń Authorization: Bearer (wysyłany przez OpenAI SDK)
            request.headers.pop("authorization", None)
            # Dodaj właściwy nagłówek Vertex AI
            request.headers["x-goog-api-key"] = self._vertex_api_key
            return super().send(request, *args, **kwargs)

    http_client = _VertexAIHttpClient(vertex_api_key=api_key)
    raw_client = OpenAI(
        api_key="dummy",  # Wymagane przez SDK, ale nadpisane przez httpx
        base_url=base_url,
        http_client=http_client,
    )
    selected_mode = mode or instructor.Mode.MD_JSON
    return instructor.patch(raw_client, mode=selected_mode)


def _create_anthropic_client(
    api_key: str,
    mode: Optional[instructor.Mode],
) -> instructor.Instructor:
    """Tworzy Anthropic client."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Biblioteka 'anthropic' nie jest zainstalowana.\n"
            "Zainstaluj: pip install anthropic"
        )

    raw_client = anthropic.Anthropic(api_key=api_key)
    selected_mode = mode or instructor.Mode.ANTHROPIC_TOOLS
    return instructor.from_anthropic(raw_client, mode=selected_mode)


# ---------------------------------------------------------------------------
# Wygodny helper: konfiguracja z env vars (opcjonalne)
# ---------------------------------------------------------------------------

def create_client_from_env(provider: LLMProvider) -> instructor.Instructor:
    """
    Tworzy klienta czytając dane z zmiennych środowiskowych.

    Zmienne środowiskowe per provider:
        OPENAI:       OPENAI_API_KEY
        ANTHROPIC:    ANTHROPIC_API_KEY
        VERTEX_AI:    VERTEX_AI_API_KEY, VERTEX_AI_BASE_URL
        AZURE_OPENAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL
    """
    import os

    env_map = {
        LLMProvider.OPENAI: {
            "api_key_var": "OPENAI_API_KEY",
            "base_url_var": None,
        },
        LLMProvider.ANTHROPIC: {
            "api_key_var": "ANTHROPIC_API_KEY",
            "base_url_var": None,
        },
        LLMProvider.VERTEX_AI: {
            "api_key_var": "VERTEX_AI_API_KEY",
            "base_url_var": "VERTEX_AI_BASE_URL",
        },
        LLMProvider.AZURE_OPENAI: {
            "api_key_var": "AZURE_OPENAI_API_KEY",
            "base_url_var": "AZURE_OPENAI_BASE_URL",
        },
    }

    config = env_map[provider]
    api_key = os.environ.get(config["api_key_var"], "")
    if not api_key:
        raise EnvironmentError(
            f"Brak zmiennej środowiskowej: {config['api_key_var']}"
        )

    base_url = None
    if config["base_url_var"]:
        base_url = os.environ.get(config["base_url_var"])

    return create_client(provider=provider, api_key=api_key, base_url=base_url)

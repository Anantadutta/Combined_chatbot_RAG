"""Thin wrapper around the Google Gen AI SDK (`google-genai`) with retries and model fallback.

- 429 (rate limit) / 5xx / timeouts are retried with exponential backoff.
- If the primary model is unavailable (404) or stays rate-limited, the fallback model is used.
"""
from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

RETRIABLE_CODES = {408, 429, 500, 502, 503, 504}
EMBED_BATCH = 100


class LLMError(RuntimeError):
    pass


def _error_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _is_retriable(exc: Exception) -> bool:
    code = _error_code(exc)
    if code in RETRIABLE_CODES:
        return True
    name = type(exc).__name__.lower()
    return any(w in name for w in ("timeout", "connect", "remoteprotocol"))


def _retry_after(exc: Exception) -> float | None:
    """Gemini 429 errors usually say 'retry in 12.3s' / 'retryDelay': '12s'."""
    m = re.search(r"retry(?:Delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)s", str(exc), re.I)
    return float(m.group(1)) if m else None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        timeout_s: int = 60,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
    ):
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self._client = client
        self.model = model
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        self.max_retries = max_retries
        self._sleep = sleep
        self.last_model_used: str | None = None

    # ------------------------------------------------------------ retry core
    def _with_retries(self, fn: Callable[[], Any], what: str, max_retries: int | None = None) -> Any:
        max_retries = self.max_retries if max_retries is None else max_retries
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt >= max_retries or not _is_retriable(exc):
                    raise
                wait = min(_retry_after(exc) or delay, 60.0) + random.uniform(0, 0.5)
                log.warning("%s failed (%s); retry %d/%d in %.1fs", what, _error_code(exc) or type(exc).__name__,
                            attempt + 1, max_retries, wait)
                self._sleep(wait)
                delay = min(delay * 2, 30.0)
        raise LLMError(f"{what}: retries exhausted")  # pragma: no cover

    # ------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], task_type: str, model: str, dim: int) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
            try:
                result = self._with_retries(
                    lambda: self._client.models.embed_content(model=model, contents=batch, config=config),
                    f"embed[{model}]",
                )
            except Exception as exc:
                raise LLMError(f"Gemini embedding failed: {exc}") from exc
            got = [list(e.values) for e in result.embeddings]
            if len(got) != len(batch):
                raise LLMError(f"Gemini returned {len(got)} embeddings for {len(batch)} texts")
            vectors.extend(got)
        return vectors

    # ------------------------------------------------------------ generation
    def generate(self, system: str, prompt: str, temperature: float = 0.1, json_mode: bool = True,
                 max_output_tokens: int = 2048) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last_exc: Exception | None = None
        for model in models:
            # When a fallback exists, give the primary model only one quick retry: a 500/503 usually
            # means Google's servers for that model are overloaded, so switching model is faster.
            retries = 1 if model != models[-1] else None
            try:
                response = self._with_retries(
                    lambda: self._client.models.generate_content(model=model, contents=prompt, config=config),
                    f"generate[{model}]",
                    max_retries=retries,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise LLMError(f"{model} returned an empty response (blocked or truncated)")
                self.last_model_used = model
                return text
            except Exception as exc:
                last_exc = exc
                code = _error_code(exc)
                if model != models[-1] and (code in (404, 429, 500, 502, 503, 504) or _is_retriable(exc)
                                            or isinstance(exc, LLMError)):
                    log.warning("model %s unavailable (%s) - falling back to %s", model, code or exc, models[-1])
                    continue
                break
        raise LLMError(f"Gemini generation failed: {last_exc}") from last_exc

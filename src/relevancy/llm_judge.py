"""Grade (query, result) pairs with an LLM on OpenRouter, Jev style.

Jev answers a relevance question with a probability for each grade instead
of free text. A small instruct model can do the same: ask for a single digit,
then read the probability of "0" through "3" from the first token's logprobs.
The expected grade (sum of grade x probability) is a continuous score that
works for reranking, and the most likely grade works as a label.
"""

import math
import os
import random
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from relevancy.rubric import Rubric
from relevancy.warehouse import ROOT

load_dotenv(ROOT / ".env")

URL = "https://openrouter.ai/api/v1/chat/completions"


GRADES = ("0", "1", "2", "3")

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
ATTEMPTS = 6
MAX_SLEEP = 30.0


@dataclass
class Grade:
    grade: int | None          # most likely grade
    prob: float | None         # its probability
    expected: float | None     # sum of grade x probability
    latency_ms: float
    cost: float | None         # USD, as OpenRouter reports it
    provider: str | None
    error: str | None = None   # set when every attempt failed; the other fields are then empty
    probs: dict[int, float] | None = None  # probability of each grade, when the judge gives one


def client() -> httpx.Client:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Add it to .env.")
    return httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60)


def _failed(error: str | None) -> Grade:
    return Grade(None, None, None, float("nan"), None, None, error=error or "unknown error")


def _post(http: httpx.Client, body: dict) -> tuple[dict | None, float, str | None]:
    """POST to OpenRouter with retries. Returns (response JSON or None, latency ms, last error)."""
    data, latency, error = None, float("nan"), None
    for attempt in range(ATTEMPTS):
        retry_after = None
        started = time.perf_counter()
        try:
            resp = http.post(URL, json=body)
            latency = (time.perf_counter() - started) * 1000  # the request only, not the backoff
            if resp.status_code in RETRY_STATUS:
                error = f"HTTP {resp.status_code}"
                try:
                    retry_after = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    pass
            else:
                resp.raise_for_status()
                data = resp.json()
                if data.get("choices"):
                    break
                # OpenRouter sometimes answers 200 with {"error": ...} when an upstream provider fails.
                error = f"no choices: {str(data.get('error', data))[:150]}"
                data = None
        except (httpx.TimeoutException, httpx.TransportError) as e:
            error = f"{type(e).__name__}: {e}"[:200]
        except (httpx.HTTPStatusError, ValueError) as e:
            if isinstance(e, httpx.HTTPStatusError):  # a 4xx that retrying won't fix
                return None, latency, f"HTTP {e.response.status_code}"
            error = f"bad JSON: {e}"[:200]
        if attempt < ATTEMPTS - 1:
            delay = retry_after if retry_after is not None else 2 ** attempt * random.uniform(0.5, 1.5)
            time.sleep(max(0.0, min(delay, MAX_SLEEP)))
    return data, latency, error


def grade(http: httpx.Client, model: str, query: str, product: str, rubric: Rubric) -> Grade:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": rubric.llm_prompt},
            {"role": "user", "content": rubric.llm_user(query, product)},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
        # Only route to providers that actually return logprobs.
        "provider": {"require_parameters": True},
        "usage": {"include": True},
    }
    data, latency, error = _post(http, body)
    if data is None:
        return _failed(error)

    choice = data["choices"][0]
    probs = dict.fromkeys(GRADES, 0.0)
    top = ((choice.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
    for t in top:
        tok = t["token"].strip()
        if tok in probs:
            probs[tok] += math.exp(t["logprob"])

    total = sum(probs.values())
    if total > 0:
        best = max(probs, key=probs.get)
        g, p = int(best), probs[best] / total
        expected = sum(int(k) * v for k, v in probs.items()) / total
    else:
        # No usable logprobs: fall back to the text, with no probability.
        text = (choice["message"].get("content") or "").strip()[:1]
        g = int(text) if text in GRADES else None
        p, expected = None, (float(g) if g is not None else None)

    return Grade(g, p, expected, latency, (data.get("usage") or {}).get("cost"), data.get("provider"))


def complete(http: httpx.Client, model: str, system: str, user: str, max_tokens: int = 400,
             json_mode: bool = False) -> str | None:
    """One chat completion's text, or None if every attempt failed. For writing text
    (rewrites, expansions) or JSON (extracted constraints, labels), not for grading."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "usage": {"include": True},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data, _, _ = _post(http, body)
    if data is None:
        return None
    return (data["choices"][0]["message"].get("content") or "").strip()

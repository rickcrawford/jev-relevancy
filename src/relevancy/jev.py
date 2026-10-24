"""Grade (query, result) pairs with Jev through TypeSafe's API.

Jev takes a `state` and typed questions and returns typed answers. A `Score`
question with the rubric's four levels, in grade order, returns a probability
for each level, so its legend index lines up with the warehouse grades (0 to 3).

All products for one query go in one request, one question each. On a probe
that cost 119 ms for four products against 134 ms for one, so reranking 50
candidates is one request per query rather than fifty.
"""

import os
import time

from dotenv import load_dotenv
from typesafe_sdk import Score, TypeSafeClient

from relevancy.llm_judge import Grade
from relevancy.rubric import Rubric
from relevancy.warehouse import ROOT

load_dotenv(ROOT / ".env")

# 1.13.0 is the version turbopuffer hosts and the one Elastic tested on ESCI.
DEFAULT_MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")


PER_REQUEST = 50  # well under Jev's 64k-token request limit at ~150 tokens a product


def client() -> TypeSafeClient:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is not set. Add it to .env.")
    return TypeSafeClient()


def grade_products(jev: TypeSafeClient, query: str, products: dict[str, str], rubric: Rubric,
                   model: str = DEFAULT_MODEL) -> dict[str, Grade]:
    """Grade every result against one query. Keys of `products` are doc ids."""
    out: dict[str, Grade] = {}
    items = list(products.items())
    for i in range(0, len(items), PER_REQUEST):
        chunk = items[i:i + PER_REQUEST]
        # Question names must be simple keys, so number them and map back.
        questions = {
            f"p{n}": Score(instructions=rubric.jev_question(text), criteria=list(rubric.levels))
            for n, (_, text) in enumerate(chunk)
        }
        started = time.perf_counter()
        resp = jev.system_one(state=rubric.jev_state(query), questions=questions, model=model)
        latency = (time.perf_counter() - started) * 1000
        data = resp.model_dump()
        for n, (doc_id, _) in enumerate(chunk):
            ans = data["answers"][f"p{n}"]
            probs = {int(k): v for k, v in ans["probabilities"].items()}
            best = max(probs, key=probs.get)
            out[doc_id] = Grade(
                grade=best, prob=probs[best], expected=ans["score"],
                latency_ms=latency, cost=None, provider=data.get("model"),
            )
    return out

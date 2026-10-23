"""Run any judge over (query, product) pairs.

A judge name says which grader to use:
  jev:jev-1.13.0          Jev via TypeSafe's API, one request per pair
  jev-batched:jev-1.13.0  Jev with all of a query's pairs in one request
  <openrouter model>      an LLM read through logprobs, e.g. openai/gpt-4o-mini
"""

import math
from concurrent.futures import ThreadPoolExecutor

from relevancy import jev, llm_judge
from relevancy.llm_judge import Grade

# A pair is (query_id, doc_id, query_text, product_text, ...); extra fields are ignored.
Pair = tuple


PREFIXES = ("jev-batched:", "jev:", "llm:")


def _failed(e: Exception) -> Grade:
    return Grade(None, None, None, math.nan, None, None, error=str(e)[:200])


def _split(judge: str) -> tuple[str, str]:
    """Return (kind, model). Only the known prefixes are stripped, so "x/y:free" stays whole."""
    for prefix in PREFIXES:
        if judge.startswith(prefix):
            return prefix[:-1], judge[len(prefix):]
    return "llm", judge


def model_of(judge: str) -> str:
    """The underlying model id for any judge string."""
    return _split(judge)[1]


def grade_pairs(judge: str, pairs: list[Pair], workers: int = 8) -> list[Grade]:
    """Return one Grade per pair, in the same order."""
    kind, model = _split(judge)
    if kind == "jev":
        client = jev.client()

        def one(p):
            try:
                return jev.grade_products(client, p[2], {p[1]: p[3]}, model=model)[p[1]]
            except Exception as e:  # noqa: BLE001  (one bad pair must not sink the batch)
                return _failed(e)

        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(one, pairs))

    if kind == "jev-batched":
        client = jev.client()
        by_query: dict[str, dict[str, str]] = {}
        text_of: dict[str, str] = {}
        for p in pairs:
            by_query.setdefault(p[0], {})[p[1]] = p[3]
            text_of[p[0]] = p[2]

        def batch(qid):
            try:
                return jev.grade_products(client, text_of[qid], by_query[qid], model=model)
            except Exception as e:  # noqa: BLE001  (one bad pair must not sink the batch)
                return _failed(e)

        with ThreadPoolExecutor(workers) as pool:
            graded = dict(zip(by_query, pool.map(batch, by_query)))
        return [g if isinstance(g := graded[p[0]], Grade) else g[p[1]] for p in pairs]

    http = llm_judge.client()

    def llm(p):
        try:
            return llm_judge.grade(http, model, p[2], p[3])
        except Exception as e:  # noqa: BLE001
            return _failed(e)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(llm, pairs))

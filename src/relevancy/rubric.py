"""Rubrics: what a judge is told "relevant" means.

A rubric lives in rubrics/<name>.toml and each dataset names one in the
`datasets` table. Jev and LLM judges both grade against it, so changing a
rubric changes every judge the same way.
"""

import tomllib
from dataclasses import dataclass

from relevancy.warehouse import ROOT

RUBRICS = ROOT / "rubrics"
LEVELS = 4  # grades 0 to 3; metrics.py counts 2 and up as relevant


@dataclass(frozen=True)
class Rubric:
    name: str
    state: str           # Jev context, with {query}
    question: str        # Jev question, followed by "<item>: <text>"
    item: str            # what a result is called: Product, Passage, Complaint, ...
    levels: tuple[str, ...]  # lowest grade first
    llm_prompt: str      # LLM system prompt

    def jev_state(self, query: str) -> str:
        return self.state.format(query=query)

    def jev_question(self, text: str) -> str:
        return f"{self.question}\n\n{self.item}: {text}"

    def llm_user(self, query: str, text: str) -> str:
        return f"Query: {query}\n\n{self.item}: {text}\n\nGrade:"


def load(name: str) -> Rubric:
    path = RUBRICS / f"{name}.toml"
    if not path.exists():
        options = ", ".join(sorted(p.stem for p in RUBRICS.glob("*.toml")))
        raise SystemExit(f"No rubric {path.relative_to(ROOT)}. Available: {options}")
    raw = tomllib.loads(path.read_text())
    levels = tuple(raw["levels"])
    if len(levels) != LEVELS:
        raise SystemExit(f"{path.name}: needs exactly {LEVELS} levels, lowest first; it has {len(levels)}")
    prompt = raw.get("llm_prompt") or "\n".join([
        f"You judge how well a {raw['item'].lower()} matches a search query.",
        "Reply with one digit and nothing else.",
        "",
        *(f"{g} = {levels[g]}" for g in reversed(range(LEVELS))),
    ])
    return Rubric(name, raw["state"], raw["question"], raw["item"], levels, prompt)


def for_dataset(con, dataset: str) -> Rubric:
    """The rubric a dataset was registered with (generic if it wasn't)."""
    row = con.execute("SELECT rubric FROM datasets WHERE dataset = ?", [dataset]).fetchone()
    return load(row[0] if row else "generic")

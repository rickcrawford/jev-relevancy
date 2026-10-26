"""Rubrics: what a judge is told "relevant" means.

A rubric lives in rubrics/<name>.toml and each dataset names one in the
`datasets` table. Jev and LLM judges both grade against it, so changing a
rubric changes every judge the same way.
"""

import tomllib
from dataclasses import dataclass

from relevancy.warehouse import ROOT

RUBRICS = ROOT / "rubrics"
GRADES = range(4)  # warehouse grades 0 to 3; metrics.py counts 2 and up as relevant


@dataclass(frozen=True)
class Rubric:
    name: str
    state: str           # Jev context, with {query}
    question: str        # Jev question, followed by "<item>: <text>"
    item: str            # what a result is called: Product, Passage, Complaint, ...
    levels: tuple[str, ...]  # lowest grade first
    llm_prompt: str      # LLM system prompt
    grades: tuple[int, ...] = (0, 1, 2, 3)  # the warehouse grade each level stands for

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
    # Four levels map to grades 0 to 3. A scale with fewer levels says which grade each one is,
    # e.g. WANDS's Irrelevant, Partial, Exact as grades = [0, 2, 3].
    if len(levels) < 4 and "grades" not in raw:
        raise SystemExit(f"{path.name}: has {len(levels)} levels, so it needs a `grades` list saying which "
                         "0-to-3 grade each level is, e.g. grades = [0, 2, 3]")
    grades = tuple(raw.get("grades", range(len(levels))))
    if not 2 <= len(levels) <= 4 or len(grades) != len(levels) or list(grades) != sorted(set(grades)) \
            or not set(grades) <= set(GRADES):
        raise SystemExit(f"{path.name}: needs 2 to 4 levels, lowest first, and if there are fewer than 4, "
                         "a `grades` list of increasing grades from 0 to 3, one per level")
    prompt = raw.get("llm_prompt") or "\n".join([
        f"You judge how well a {raw['item'].lower()} matches a search query.",
        "Reply with one digit and nothing else.",
        "",
        *(f"{g} = {level}" for g, level in reversed(list(zip(grades, levels)))),
    ])
    return Rubric(name, raw["state"], raw["question"], raw["item"], levels, prompt, grades)


def for_dataset(con, dataset: str) -> Rubric:
    """The rubric a dataset was registered with (generic if it wasn't)."""
    row = con.execute("SELECT rubric FROM datasets WHERE dataset = ?", [dataset]).fetchone()
    return load(row[0] if row else "generic")

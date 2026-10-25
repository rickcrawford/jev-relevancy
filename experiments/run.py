"""Run the studies in experiments/ from one place, by hand or from an agent.

Each study folder has a study.toml: its premise, status, the keys and files it
needs, whether it needs the ESCI base, its steps in order, and the result that
would count. This reads those manifests and runs the steps.

  uv run experiments/run.py list                     every study, ranked
  uv run experiments/run.py show recall-fusion       premise, steps, pass mark
  uv run experiments/run.py check recall-fusion      keys, base, files: what's missing
  uv run experiments/run.py base                     what the ESCI base still needs
  uv run experiments/run.py base --yes               build it (paid: ~80M embedding tokens)
  uv run experiments/run.py run recall-fusion        dry run: print the steps
  uv run experiments/run.py run recall-fusion --yes  run them
  uv run experiments/run.py run recall-fusion --yes --step diagnose
  uv run experiments/run.py status                   last run of every study

Add --json to list, show, check, base, or status for output an agent can parse.

Steps that cost money (turbopuffer queries, Jev, an LLM) only run with --yes.
Without it, `run` prints what it would do. Each step's output goes to the
study's results/ folder, and every run writes results/runs/<time>.json with
the steps, exit codes, durations, and git commit, so a result traces back.
The runner never changes a study's pass mark or README; recording the result
is a separate, deliberate step (see .claude/skills/run-study/SKILL.md).
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
load_dotenv(ROOT / ".env")

# The ESCI base every study needs: what each piece is, how to check it, and how to build it.
BASE = [
    ("ESCI downloaded", lambda con: (ROOT / "data" / "raw" / "esci" / "shopping_queries_dataset_products.parquet").exists(),
     "./scripts/01_download_esci.sh", False),
    ("500 sampled queries", lambda con: con.execute(
        "SELECT count(*) FROM queries WHERE dataset = 'esci-us' AND sampled").fetchone()[0] >= 500,
     "uv run scripts/02_prepare_esci.py", False),
    ("full namespace loaded and searched (hybrid-rrf on esci-us-full)", lambda con: _run_exists(
        con, "retriever = 'hybrid-rrf' AND reranker = 'none'"),
     "uv run scripts/03_load_turbopuffer.py --scope full && uv run scripts/04_search.py --scope full", True),
    ("Voyage rerank of hybrid", lambda con: _run_exists(con, "reranker = 'voyage/rerank-3'"),
     "uv run scripts/06_rerank.py --model voyage/rerank-3", True),
    ("Jev rerank of hybrid", lambda con: _run_exists(con, "reranker = 'jev:jev-1.13.0'"),
     "uv run scripts/06_rerank.py --model jev:jev-1.13.0", True),
    ("Jev grades for every run's top 10 (step 8)", lambda con: con.execute(
        "SELECT count(*) FROM judgments WHERE dataset = 'esci-us' AND source = 'jev-batched:jev-1.13.0'"
    ).fetchone()[0] > 0,
     "uv run scripts/08_judge_runs.py --judge jev-batched:jev-1.13.0", True),
]


def _run_exists(con, where: str) -> bool:
    return con.execute(f"""SELECT count(*) FROM runs WHERE dataset = 'esci-us'
                           AND config->>'namespace' = 'esci-us-full' AND {where}""").fetchone()[0] > 0


def studies() -> dict[str, dict]:
    out = {}
    for path in sorted(EXP.glob("*/study.toml")):
        s = tomllib.loads(path.read_text())
        s["folder"] = path.parent.name
        out[s["folder"]] = s
    return dict(sorted(out.items(), key=lambda kv: (kv[1].get("rank") or 99, kv[0])))


def get(name: str) -> dict:
    all_ = studies()
    if name not in all_:
        raise SystemExit(f"No study {name!r}. Try: uv run experiments/run.py list")
    return all_[name]


def git_sha() -> str:
    def git(*a):
        return subprocess.run(["git", *a], check=False, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD")
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


# -- checks -------------------------------------------------------------------

def base_status() -> list[dict]:
    db = Path(os.environ.get("RELEVANCY_DB", ROOT / "warehouse" / "relevancy.duckdb"))
    if not db.exists():
        return [{"item": what, "ok": i == 0 and test(None), "build": cmd, "paid": paid}
                for i, (what, test, cmd, paid) in enumerate(BASE)]
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    out = []
    for what, test, cmd, paid in BASE:
        try:
            ok = bool(test(con))
        except duckdb.Error:
            ok = False
        out.append({"item": what, "ok": ok, "build": cmd, "paid": paid})
    con.close()
    return out


def check(s: dict, scope: str | None = None) -> dict:
    keys = {k: bool(os.environ.get(k)) for k in s.get("keys", [])}
    files = {f: (ROOT / f).exists() for f in s.get("files", [])}
    # A smoke-scope pilot runs on the quick start's namespace, not the full base.
    base = base_status() if s.get("base", True) and scope != "smoke" else []
    missing = ([f"env {k}" for k, ok in keys.items() if not ok]
               + [f"file {f}" for f, ok in files.items() if not ok]
               + [f"base: {b['item']}" for b in base if not b["ok"]])
    if s["status"] == "planned":
        missing.append("no script yet: implement it from the README first")
    return {"study": s["folder"], "ready": not missing, "missing": missing, "keys": keys, "files": files,
            "base": base}


# -- running ------------------------------------------------------------------

def run_steps(s: dict, steps: list[dict], yes: bool, label: str) -> int:
    results = EXP / s["folder"] / s.get("results", "results")
    record = {"study": s["folder"], "label": label, "commit": git_sha(),
              "started": datetime.now(UTC).isoformat(timespec="seconds"), "steps": []}
    paid = [st["name"] for st in steps if st.get("paid", True)]
    if not yes:
        print(f"Dry run of {s['folder']}. Add --yes to run" + (f"; paid steps: {', '.join(paid)}" if paid else "") + ".")
        for st in steps:
            print(f"  [{st['name']}] {'paid ' if st.get('paid', True) else 'free '} {st['run']}"
                  + (f"  > {st['output']}" if st.get("output") else ""))
        return 0
    for st in steps:
        print(f"\n== {s['folder']}: {st['name']}\n$ {st['run']}", flush=True)
        started = time.monotonic()
        out_path = (EXP / s["folder"] / st["output"]) if st.get("output") else results / "logs" / f"{st['name']}.log"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as log:
            proc = subprocess.Popen(st["run"], shell=True, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        record["steps"].append({"name": st["name"], "run": st["run"], "exit": code,
                                "seconds": round(time.monotonic() - started, 1),
                                "output": str(out_path.relative_to(ROOT))})
        if code:
            print(f"\nStep {st['name']} failed with exit code {code}. Later steps didn't run.")
            break
    record["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["ok"] = all(st["exit"] == 0 for st in record["steps"]) and len(record["steps"]) == len(steps)
    runs = results / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{record['started'].replace(':', '')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nRun record: {path.relative_to(ROOT)}")
    if record["ok"] and s.get("decide"):
        print(f"\nNow compare the output with the pass mark (README, \"What counts\"):\n  {s['decide']}")
    return 0 if record["ok"] else 1


def last_run(folder: str) -> dict | None:
    runs = sorted((EXP / folder / "results" / "runs").glob("*.json"))
    return json.loads(runs[-1].read_text()) if runs else None


# -- commands -----------------------------------------------------------------

def cmd_list(args):
    rows = [{"study": k, "rank": s.get("rank"), "status": s["status"], "premise": s["premise"]}
            for k, s in studies().items()]
    if args.json:
        return print(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['rank'] or '':>3}  {r['study']:<30} {r['status']:<8} {r['premise']}")


def cmd_show(args):
    s = get(args.study)
    if args.json:
        return print(json.dumps(s, indent=2))
    print(f"{s['folder']}: {s['title']}\n\nPremise: {s['premise']}\nStatus: {s['status']}"
          f"\nKeys: {', '.join(s.get('keys', [])) or 'none'}\nNeeds the ESCI base: {s.get('base', True)}")
    if s.get("files"):
        print(f"Files: {', '.join(s['files'])}")
    print("\nSteps:" if s.get("steps") else "\nSteps: none yet (planned). The README says what to build.")
    for st in s.get("steps", []):
        print(f"  {st['name']:<12} {'paid' if st.get('paid', True) else 'free'}  {st['run']}")
    print(f"\nWhat counts: {s.get('decide', 'see README')}\nREADME: experiments/{s['folder']}/README.md")


def cmd_check(args):
    targets = [get(args.study)] if args.study else list(studies().values())
    results = [check(s) for s in targets]
    if args.json:
        print(json.dumps(results if not args.study else results[0], indent=2))
    else:
        for r in results:
            print(f"{r['study']:<30} {'ready' if r['ready'] else 'missing: ' + '; '.join(r['missing'])}")
    return 0 if all(r["ready"] for r in results) else 1


def cmd_base(args):
    status = base_status()
    if args.json and not args.yes:
        return print(json.dumps(status, indent=2))
    todo = [b for b in status if not b["ok"]]
    for b in status:
        print(f"  {'ok  ' if b['ok'] else 'TODO'} {b['item']}" + ("" if b["ok"] else f"\n       {b['build']}"))
    if not todo:
        return print("The base is built.")
    steps = [{"name": f"base-{i}", "run": b["build"], "paid": b["paid"]} for i, b in enumerate(todo, 1)]
    return run_steps({"folder": "records", "results": ".", "decide": ""}, steps, args.yes, "base")


def cmd_run(args):
    s = get(args.study)
    if s["status"] == "planned" or not s.get("steps"):
        raise SystemExit(f"{s['folder']} has no script yet. Build it from experiments/{s['folder']}/README.md "
                         "and add its steps to study.toml.")
    steps = s["steps"]
    names = [st["name"] for st in steps]
    if args.step:
        unknown = set(args.step) - set(names)
        if unknown:
            raise SystemExit(f"Unknown step(s) {', '.join(sorted(unknown))}. Steps: {', '.join(names)}")
        steps = [st for st in steps if st["name"] in args.step]
    elif args.from_step:
        if args.from_step not in names:
            raise SystemExit(f"Unknown step {args.from_step}. Steps: {', '.join(names)}")
        steps = steps[names.index(args.from_step):]
    if args.scope:
        steps = [st | {"run": f"{st['run']} --scope {shlex.quote(args.scope)}"} if st.get("scoped") else st
                 for st in steps]
    if args.yes:
        c = check(s, args.scope)
        if not c["ready"]:
            raise SystemExit("Not ready: " + "; ".join(c["missing"]))
    return run_steps(s, steps, args.yes, "run")


def cmd_status(args):
    rows = []
    for k, s in studies().items():
        r = last_run(k)
        rows.append({"study": k, "status": s["status"], "last_run": r and r["started"], "ok": r and r["ok"],
                     "commit": r and r["commit"]})
    if args.json:
        return print(json.dumps(rows, indent=2))
    for r in rows:
        last = f"{r['last_run']} {'ok' if r['ok'] else 'FAILED'} at {r['commit']}" if r["last_run"] else "never run here"
        print(f"{r['study']:<30} {r['status']:<8} {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("list", "status"):
        p = sub.add_parser(name)
        p.add_argument("--json", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("study")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("study", nargs="?")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("base")
    p.add_argument("--yes", action="store_true", help="build what's missing (paid)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("study")
    p.add_argument("--yes", action="store_true", help="actually run; without it, print the steps")
    p.add_argument("--step", action="append", help="run only this step (repeatable)")
    p.add_argument("--from", dest="from_step", help="start at this step")
    p.add_argument("--scope", choices=["smoke", "full"], help="for studies whose steps take --scope")
    args = ap.parse_args()
    handler = {"list": cmd_list, "show": cmd_show, "check": cmd_check, "base": cmd_base,
               "run": cmd_run, "status": cmd_status}[args.command]
    return handler(args) or 0


if __name__ == "__main__":
    sys.exit(main())

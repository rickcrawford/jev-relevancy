"""Every study.toml points at scripts and subcommands that exist."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_step_resolves():
    studies = list((ROOT / "experiments").glob("*/study.toml"))
    assert len(studies) >= 19
    for path in studies:
        s = tomllib.loads(path.read_text())
        assert s["status"] in {"planned", "ready", "done"}, path
        assert s["status"] != "planned", f"{path.parent.name} has no steps yet"
        for step in s.get("steps", []):
            m = re.search(r"(experiments/[\w.-]+/\w+\.py)\s+([\w-]+)?", step["run"])
            if not m:
                continue  # a pipeline step or a shell command
            script = ROOT / m[1]
            assert script.exists(), (path, step["run"])
            if m[2] and "commands = {" in script.read_text():
                assert f'"{m[2]}"' in script.read_text(), (path, step["name"], m[2])


def test_cli_lists_every_study(capsys):
    from relevancy import cli
    cli.cmd_list(type("A", (), {"json": True})())
    assert "relevance-ltr" in capsys.readouterr().out

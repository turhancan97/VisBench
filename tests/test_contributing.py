"""The contributor guide has to stay true.

A setup document that drifts from what CI actually runs is worse than none: it
sends someone to run the wrong commands, watch them pass, and then fail review
for a reason the guide told them not to expect.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "CONTRIBUTING.md"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_the_documented_lint_commands_are_the_ones_ci_runs():
    """Copied by hand into the guide, so pinned here rather than trusted."""
    guide = GUIDE.read_text()
    workflow = WORKFLOW.read_text()

    for command in (
        "ruff check visbench/ tests/ conftest.py examples/ scripts/",
        "ruff format --check visbench/ tests/ conftest.py examples/ scripts/",
        "mypy visbench/ examples/ --ignore-missing-imports",
    ):
        assert command in workflow, f"CI no longer runs: {command}"
        assert command in guide, f"CONTRIBUTING.md does not document: {command}"


def test_every_ci_job_is_accounted_for():
    """A new gating job that nobody is told about fails a first PR silently."""
    jobs = set(re.findall(r"^  ([a-z_]+):$", WORKFLOW.read_text(), flags=re.M))
    jobs -= {"push", "pull_request", "workflow_dispatch", "schedule"}
    guide = GUIDE.read_text()

    documented = {
        "lint": "ruff check",
        "test": "pytest",
        "lock": "uv lock --check",
        "build": "twine check",
    }
    assert jobs == set(documented), f"CI jobs changed: {sorted(jobs)}"
    for job, evidence in documented.items():
        assert evidence in guide, f"CONTRIBUTING.md never mentions the {job!r} job ({evidence})"


def test_the_extras_it_names_exist():
    """`pip install -e ".[dev,clip,timm,hub]"` must not name a missing extra."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    named = re.search(r'pip install -e "\.\[([^\]]+)\]"', GUIDE.read_text())
    assert named, "the guide no longer shows an editable install"
    for extra in named.group(1).split(","):
        assert re.search(rf"^{extra.strip()} = ", pyproject, flags=re.M), (
            f"CONTRIBUTING.md offers a '{extra.strip()}' extra that pyproject.toml does not define"
        )


def test_it_points_at_pages_that_exist():
    for target in re.findall(r"\]\((?!https?:)([^)#]+)\)", GUIDE.read_text()):
        assert (ROOT / target).exists(), f"CONTRIBUTING.md links to a missing {target}"


def test_the_issue_templates_are_valid_yaml():
    """A malformed template is silently ignored by GitHub."""
    import yaml

    templates = sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))
    assert templates, "no issue templates found"
    for path in templates:
        yaml.safe_load(path.read_text())

"""Push and pull, with the Hub faked.

Nothing here touches the network. The transport is deliberately thin — it stages
a file and hands it to ``huggingface_hub`` — so what is worth testing is not
that an upload works, but that the thin layer cannot become a second, weaker
loading path than the local one.
"""

import pytest
import torch

import visbench
from visbench.hub import (
    IncompatibleProbe,
    load_probe_from_hub,
    probe_card,
    push_probe,
    save_probe,
)
from visbench.tasks.high_level.classification import ClassificationTask


@pytest.fixture
def fitted(fake_vit):
    torch.manual_seed(0)
    features = {"pooled": torch.randn(40, fake_vit.embed_dim)}
    labels = torch.randint(0, 3, (40,))
    return ClassificationTask(epochs=2).fit(features, labels), features, labels


class FakeApi:
    """Records what it was asked to do, and writes nothing anywhere."""

    def __init__(self, token=None):
        self.token = token
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        from pathlib import Path

        folder = Path(kwargs["folder_path"])
        # Capture the staged contents: the folder is a TemporaryDirectory and
        # is gone by the time the test looks.
        kwargs["files"] = {p.name: p.read_bytes() for p in folder.iterdir()}
        self.uploaded.append(kwargs)


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr("visbench.hub.remote._api", lambda token=None: api)
    return api


class FakeHub:
    """A stand-in ``huggingface_hub`` module, recording what it was asked for."""

    def __init__(self):
        self.path: str | None = None
        self.calls: list[dict] = []

    def hf_hub_download(self, **kwargs):
        self.calls.append(kwargs)
        return self.path


@pytest.fixture
def fake_hub(monkeypatch):
    """Install a stub ``huggingface_hub`` in ``sys.modules``.

    Injected rather than monkeypatched onto the real package, because **CI
    installs `.[dev]` only** — no clip, no timm, no hub — so the real module is
    absent there and ``monkeypatch.setattr("huggingface_hub....")`` raises
    ModuleNotFoundError at collection. It passed locally because this venv has
    huggingface_hub transitively via timm, which is exactly the "a local env
    with extra packages passes checks CI fails" trap.

    Skipping instead would have left the *pull* path untested in a core
    install, and pull is the half that loads someone else's file.
    """
    import sys
    import types

    stub = FakeHub()
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = stub.hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return stub


# --------------------------------------------------------------------------
# Push
# --------------------------------------------------------------------------


def test_push_uploads_the_weights_and_a_card(fitted, fake_vit, fake_api):
    task, _, _ = fitted
    url = push_probe(task, "someone/a-probe", backbone=fake_vit)

    assert url == "https://huggingface.co/someone/a-probe"
    files = fake_api.uploaded[0]["files"]
    assert set(files) == {"probe.pt", "README.md"}
    assert b"visbench" in files["README.md"]


def test_push_defaults_to_private(fitted, fake_vit, fake_api):
    """A push is not reversible the way a local write is.

    Once a repository is public it may already have been fetched, and deleting
    it afterwards does not unpublish what was taken. Public is a decision, not
    a default to discover afterwards.
    """
    task, _, _ = fitted
    push_probe(task, "someone/a-probe", backbone=fake_vit)
    assert fake_api.created[0]["private"] is True


def test_public_must_be_asked_for(fitted, fake_vit, fake_api):
    task, _, _ = fitted
    push_probe(task, "someone/a-probe", backbone=fake_vit, private=False)
    assert fake_api.created[0]["private"] is False


def test_an_unfitted_probe_never_reaches_the_hub(fake_vit, fake_api):
    """The refusal has to happen before the repository is created.

    Otherwise a rejected push leaves an empty public repository behind, which is
    worse than the error it was trying to report.
    """
    task = visbench.get_probe("classification")
    with pytest.raises(ValueError, match="has not been fitted"):
        push_probe(task, "someone/a-probe", backbone=fake_vit)

    assert fake_api.created == []
    assert fake_api.uploaded == []


def test_a_zero_shot_probe_never_reaches_the_hub(fake_vit, fake_api):
    task = visbench.get_probe("retrieval")
    with pytest.raises(ValueError, match="zero-shot"):
        push_probe(task, "someone/a-probe", backbone=fake_vit)
    assert fake_api.created == []


def test_the_uploaded_file_is_the_artifact_the_local_saver_writes(
    fitted, fake_vit, fake_api, tmp_path
):
    """Push must not invent its own format.

    If it did, a pulled probe and a locally saved one would drift, and only one
    of them would be covered by the artifact tests.
    """
    task, _, _ = fitted
    push_probe(task, "someone/a-probe", backbone=fake_vit)
    uploaded = fake_api.uploaded[0]["files"]["probe.pt"]

    local = tmp_path / "probe.pt"
    save_probe(task, local, backbone=fake_vit)

    pushed = torch.load(_as_file(uploaded, tmp_path / "u.pt"), weights_only=True)
    saved = torch.load(local, weights_only=True)
    assert pushed["meta"] == saved["meta"]
    assert pushed["head_spec"] == saved["head_spec"]


def _as_file(payload: bytes, path):
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------


def test_a_pulled_probe_predicts_what_was_pushed(fitted, fake_vit, tmp_path, fake_hub):
    task, features, _ = fitted
    before = task.predict(features)

    path = tmp_path / "probe.pt"
    save_probe(task, path, backbone=fake_vit)
    fake_hub.path = str(path)

    loaded = load_probe_from_hub("someone/a-probe", backbone=fake_vit)
    assert torch.equal(before, loaded.predict(features))


def test_a_pulled_probe_gets_the_same_identity_checks(fitted, fake_vit, tmp_path, fake_hub):
    """The reason pull is `load_probe` with a download in front of it.

    A separate remote path is how one of them ends up without the backbone check
    or without ``weights_only`` — and a downloaded probe is the one that most
    needs both.
    """
    task, _, _ = fitted
    path = tmp_path / "probe.pt"
    save_probe(task, path, backbone=fake_vit)
    fake_hub.path = str(path)

    other = visbench.get_probe("classification")
    other.pooling = "mean"
    with pytest.raises(IncompatibleProbe, match="pooling"):
        load_probe_from_hub("someone/a-probe", backbone=fake_vit, task=other)


def test_a_revision_is_passed_through(fitted, fake_vit, tmp_path, fake_hub):
    """A Hub repo is mutable; `main` today is not promised to be `main` later."""
    task, _, _ = fitted
    path = tmp_path / "probe.pt"
    save_probe(task, path, backbone=fake_vit)
    fake_hub.path = str(path)

    load_probe_from_hub("someone/a-probe", backbone=fake_vit, revision="abc123")

    assert fake_hub.calls[0]["revision"] == "abc123"
    assert fake_hub.calls[0]["filename"] == "probe.pt"


# --------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------


def test_the_card_names_the_backbone_the_probe_belongs_to(fitted, fake_vit):
    """A bare .pt on a model page tells a visitor nothing.

    Least of all the one thing they must know: that these weights belong to
    exactly one backbone.
    """
    task, _, _ = fitted
    card = probe_card(task, fake_vit, "someone/a-probe")

    assert fake_vit.name in card
    assert fake_vit.cache_key() in card
    assert "classification" in card
    assert "load_probe_from_hub" in card
    assert "someone/a-probe" in card


def test_the_card_records_resolved_pooling(fitted, fake_vit):
    task, _, _ = fitted
    card = probe_card(task, fake_vit, "someone/a-probe")
    assert "`cls` (requested `default`)" in card


def test_the_card_reports_metrics_when_given_them(fitted, fake_vit):
    task, _, _ = fitted
    card = probe_card(task, fake_vit, "someone/a-probe", metrics={"top1": 0.982})
    assert "| `top1` | 0.9820 |" in card


def test_the_card_omits_the_metrics_table_when_there_are_none(fitted, fake_vit):
    task, _, _ = fitted
    assert "Reported scores" not in probe_card(task, fake_vit, "someone/a-probe")


def test_the_card_front_matter_comes_first(fitted, fake_vit):
    """The Hub parses YAML front matter only at the very start of the file."""
    task, _, _ = fitted
    card = probe_card(task, fake_vit, "someone/a-probe")
    assert card.startswith("---\n")
    assert card.count("---\n") >= 2
    assert "library_name: visbench" in card.split("---\n")[1]


def test_the_card_and_the_artifact_cannot_disagree(fitted, fake_vit, tmp_path):
    """Both are generated from `probe_metadata`, so this pins that they stay so."""
    task, _, _ = fitted
    save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)
    meta = torch.load(tmp_path / "probe.pt", weights_only=True)["meta"]

    card = probe_card(task, fake_vit, "someone/a-probe")
    assert meta["backbone_key"] in card
    assert meta["feature_mode"] in card


# --------------------------------------------------------------------------
# The [hub] extra is optional, and that is a promise about a core install
# --------------------------------------------------------------------------


@pytest.fixture
def no_hub(monkeypatch):
    """Make ``import huggingface_hub`` fail, as it would without the extra."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)


def test_importing_the_package_does_not_need_the_extra(no_hub):
    """The import must be inside the functions, not at module scope.

    Moving it to the top would break `import visbench.hub` — and therefore
    saving a probe to a local file — for every core install.

    Both modules are reloaded, and ``remote`` is the one that matters: it holds
    the only ``huggingface_hub`` references. Reloading just ``visbench.hub``
    proves nothing, because ``remote`` is already in ``sys.modules`` by then and
    a module-scope import there would never be re-executed. Written the weak way
    first, and it passed with the import moved to the top of the file.
    """
    import importlib

    import visbench.hub
    import visbench.hub.remote

    importlib.reload(visbench.hub.remote)
    importlib.reload(visbench.hub)
    assert hasattr(visbench.hub, "save_probe")
    assert hasattr(visbench.hub, "push_probe")


def test_saving_and_loading_locally_does_not_need_the_extra(fitted, fake_vit, tmp_path, no_hub):
    from visbench.hub import load_probe

    task, features, _ = fitted
    save_probe(task, tmp_path / "probe.pt", backbone=fake_vit)
    loaded = load_probe(tmp_path / "probe.pt", backbone=fake_vit)
    assert torch.equal(task.predict(features), loaded.predict(features))


def test_pushing_without_the_extra_says_how_to_get_it(fitted, fake_vit, no_hub):
    task, _, _ = fitted
    with pytest.raises(ImportError, match=r"pip install visbench\[hub\]"):
        push_probe(task, "someone/a-probe", backbone=fake_vit)


def test_pulling_without_the_extra_says_how_to_get_it(fake_vit, no_hub):
    with pytest.raises(ImportError, match=r"pip install visbench\[hub\]"):
        load_probe_from_hub("someone/a-probe", backbone=fake_vit)

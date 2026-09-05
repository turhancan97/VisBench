# Installation

```bash
pip install visbench                    # core: DINOv2, every task, the CLI
pip install 'visbench[clip,timm]'       # + CLIP and timm backbones
pip install 'visbench[hub]'             # + push/pull probes to Hugging Face
```

`clip` and `timm` are optional extras. A backbone whose extra is missing stays
listed — `visbench list backbones` marks it — and constructing one tells you
which extra to install rather than pretending the name does not exist.

`hub` is needed only to *transfer* a probe. Saving one to a local file and
loading it back works in a core install.

## From source

```bash
git clone https://github.com/turhancan97/VisBench && cd VisBench
uv sync --all-extras            # exact locked versions — what the numbers below used
# or
pip install -e ".[dev,clip,timm]"
pytest              # fast tests, no weights downloaded
pytest -m slow      # also runs the real DINOv2 and CLIP checkpoints

# The three gating lint steps, exactly as CI runs them. Run them verbatim —
# mypy in particular reads [tool.mypy] from pyproject.toml, so invoking it
# with different flags checks something CI does not.
ruff check visbench/ tests/ conftest.py examples/
ruff format --check visbench/ tests/ conftest.py examples/
mypy visbench/ examples/ --ignore-missing-imports
```

`uv sync --all-extras` installs the **exact locked versions** every published
VisBench number was produced with. `uv.lock` pins the package itself, so a
version bump desynchronises it and CI's `lock` job fails while all five local
commands pass.

## What each extra buys

| extra | what it adds | needed for |
| --- | --- | --- |
| *(core)* | DINOv2, ResNets, every probe, the CLI | most things |
| `clip` | OpenCLIP backbones | `clip_vitb16`, `clip_vitb32` |
| `timm` | timm CNNs and ViTs | ConvNeXt, MAE, SigLIP, DINO, SAM, supervised ViT |
| `hub` | `huggingface_hub` | pushing and pulling a trained probe |
| `datasets` | Hugging Face `datasets` | `--dataset hf:...` |
| `docs` | Sphinx, Furo, MyST | building this site |
| `dev` | pytest, ruff, mypy | contributing |
| `all` | `clip,timm,hub,datasets` | the runtime set; **not** `docs` or `dev` |

`all` deliberately excludes `docs`: that extra is for building this site, and
pulling Sphinx into a weights-downloading test run is noise.

## Checking it worked

```bash
visbench demo
```

No dataset, no configuration, no large download — {doc}`what it does
</getting-started/quickstart>`.

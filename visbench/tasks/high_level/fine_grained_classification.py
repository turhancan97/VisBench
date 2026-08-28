"""Fine-grained recognition — a linear probe on pooled features, on species labels.

Mechanically identical to :class:`~visbench.tasks.high_level.classification.ClassificationTask`
and to :class:`~visbench.tasks.high_level.scene_classification.SceneClassificationTask`:
one linear layer, AdamW, top-1/top-5. What differs is the *question*, and here
the difference is one of granularity rather than of subject.

Object classification asks whether a representation separates *basic-level*
categories — a bird from a car — which is the level ImageNet-1k supervision
optimises directly and the level at which the Imagenette board is saturated.
Fine-grained recognition asks whether it separates *subordinate* categories
inside one basic-level class: two hundred species of bird that share a body
plan, a pose distribution and a background distribution, and differ in the
shape of a beak or the colour of a wing bar. A backbone can carry the first
distinction cleanly while discarding the second, which is exactly what a
probe is for — the information either survived the encoder or it did not.

**Why a distinct probe and not a second dataset for** ``classification``.
:func:`scripts.render_tables.board_for` renders exactly one table per task and
refuses a task with more than one comparability group, and
:func:`~visbench.results.leaderboard.comparability_key` groups by dataset name
and fingerprint. A CUB record under ``task="classification"`` would therefore
not join the Imagenette board — it would make that board *unrenderable*. A
separate task name gives fine-grained recognition its own board, its own CLI
row and its own leaderboard group, the same way ``scene_classification`` is
distinct from ``classification`` and ``corner`` from ``edge`` although each
pair shares every line of its implementation.

The canonical dataset is CUB-200-2011 (200 bird species, the official
5994/5794 split), which ships on this machine in the standard labelled layout
— ``train/<class>/`` + ``val/<class>/`` — and which
:class:`~visbench.data.image_folder.ImageFolderDataset` reads with no loader
code. Any folder of subordinate-category photographs runs it: Stanford Cars
and Stanford Dogs are the same shape, and would be the same probe under a
different ``dataset`` fingerprint, hence a different comparability group.

**One caveat belongs with every number this probe produces**, and it is
recorded in ``CAVEATS`` rather than left to the reader: ImageNet-1k contains
59 bird classes, several of which overlap CUB's species, so the two
ImageNet-1k-supervised backbones in the corpus (``convnext_base``,
``supervised_vitb16``) are closer to in-distribution recall here than the
self-supervised ones are. That is the same confound
``CORPUS_FINDINGS.md`` already records for the Imagenette board — weaker
here, because subordinate species labels are not what those models were
trained to emit, but not absent.
"""

from visbench.registry import register_task
from visbench.tasks.high_level.classification import ClassificationTask

__all__ = ["FineGrainedClassificationTask"]


@register_task("fine_grained_classification")
class FineGrainedClassificationTask(ClassificationTask):
    """Linear-probe classification of subordinate (species / model) categories.

    Every mechanical part — the lazily-built ``nn.Linear``, the AdamW schedule
    whose defaults were chosen by measurement, the optional standardiser and its
    ``probe_state`` serialisation — is inherited unchanged from
    :class:`ClassificationTask`. Only the identity of the number changes.

    The schedule defaults are the parent's, and that was checked rather than
    assumed. 200 classes over ~6k training images looks like the case most
    likely to underfit at ten times the class count and a fraction of the data
    — and it does not: ``train_top1`` is **1.0000** on all six backbones of the
    rank check, since a linear map from 384-2048 dimensions to 200 classes has
    enough capacity to separate 5994 points.

    That matters for how a low score here is read. The gap between 1.0 and the
    validation score is *generalisation*, not an unconverged probe, so a weak
    number is a property of the representation — which is the only condition
    under which a probe measures anything. The spread over those six is
    **0.3976** (``dinov2_vitb14`` 0.8674 to ``mae_vitb16`` 0.4698), against an
    object board whose leaders sit within a point of each other.
    :attr:`train_top1` is still the field to check first on a backbone that was
    not among the six.
    """

    level = "high_level"
    display_name = "Fine-grained recognition"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The base hardcodes self.name = "classification"; this is the one field
        # that has to move, and it is what keeps the boards apart.
        self.name = "fine_grained_classification"

    def describe(self) -> dict:
        """Task metadata plus a ``protocol`` naming this as the fine-grained probe.

        Object classification records carry no ``protocol`` key. This one does,
        purely for provenance — the task name already separates the
        comparability groups, so the string changes nothing about ranking; it
        just says in the record what kind of number this is.
        """
        described = super().describe()
        described["task_params"]["protocol"] = "visbench_fine_grained_linear_probe"
        return described

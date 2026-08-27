"""Scene classification — a linear probe on pooled features, on scene labels.

Mechanically identical to :class:`~visbench.tasks.high_level.classification.ClassificationTask`:
one linear layer, AdamW, top-1/top-5. What differs is the *question*. Object
classification asks whether a representation separates object categories;
scene classification asks whether it encodes the category of a *place* — the
global layout and context of a room or a landscape — which a backbone can be
good at while being weak at the other, and vice versa. Keeping them as two
probes rather than one dataset flag is what lets the two boards disagree
without one silently overwriting the other.

**Why a distinct probe and not a second dataset for** ``classification``.
:func:`scripts.render_tables.board_for` renders exactly one table per task and
refuses a task with more than one comparability group, and
:func:`~visbench.results.leaderboard.comparability_key` groups by dataset name
and fingerprint. A Places365 record under ``task="classification"`` would
therefore make the object-classification board unrenderable. A separate task
name gives scene classification its own board, its own CLI row and its own
leaderboard group, the same way ``corner`` is a distinct probe from ``edge``
although they share every line of their implementation.

The canonical dataset is Places365 (``train/<class>/`` + ``val/<class>/``,
365 classes), which :class:`~visbench.data.image_folder.ImageFolderDataset`
reads with no loader code. Any folder of scene photographs runs it.
"""

from visbench.registry import register_task
from visbench.tasks.high_level.classification import ClassificationTask

__all__ = ["SceneClassificationTask"]


@register_task("scene_classification")
class SceneClassificationTask(ClassificationTask):
    """Linear-probe classification of scene / place categories.

    Every mechanical part — the lazily-built ``nn.Linear``, the AdamW schedule
    whose defaults were chosen by measurement, the optional standardiser and its
    ``probe_state`` serialisation — is inherited unchanged from
    :class:`ClassificationTask`. Only the identity of the number changes.
    """

    level = "high_level"
    display_name = "Scene classification"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The base hardcodes self.name = "classification"; this is the one field
        # that has to move, and it is what keeps the two boards apart.
        self.name = "scene_classification"

    def describe(self) -> dict:
        """Task metadata plus a ``protocol`` naming this as the scene probe.

        Object classification records carry no ``protocol`` key. This one does,
        purely for provenance — the task name already separates the
        comparability groups, so the string changes nothing about ranking; it
        just says in the record what kind of number this is.
        """
        described = super().describe()
        described["task_params"]["protocol"] = "visbench_scene_linear_probe"
        return described

"""Scene parsing — dense category prediction over an indoor scene, on NYUv2-40.

Mechanically identical to
:class:`~visbench.tasks.high_level.semantic_segmentation.SemanticSegmentationTask`:
the same linear head over frozen features, the same cross-entropy with
``ignore_index``, the same two mIoUs reported under distinct names. What differs
is the *question*, and it differs in two ways at once.

**Stuff rather than things.** VOC's twenty classes are objects on a background;
NYU40's are a room — wall, floor, ceiling, and the furniture in it. Most of the
pixels in this target belong to categories that have no instances and no
boundaries in the object sense, which is what the segmentation literature calls
scene parsing rather than object segmentation.

**Forty classes rather than twenty-one**, on 795 training images rather than
1,464 — so this board asks a harder question of a smaller split, and its
absolute mIoU is not comparable with the VOC board's. Nothing in a record says
that; the task name is what keeps the two boards apart.

**Why a distinct probe and not a second dataset for** ``semantic_segmentation``.
``board_for`` renders exactly one table per task and refuses a task with more
than one comparability group, and ``comparability_key`` groups by dataset name
and fingerprint — so an NYUv2 record under ``task="semantic_segmentation"``
would make the VOC board **unrenderable**. A separate name gives this its own
board, its own CLI row and its own leaderboard group, the way
``scene_classification`` is separate from ``classification``.

**It reads the frames `depth` and `surface_normal` already read.** The
canonical 795/654 labelled split ships as stem-matched folders, so
``images/`` is shared with those two boards and only ``segmentation_nyu40/``
is new — three probes over identical pixels asking geometry, geometry and
semantics. That is the NYUv2 counterpart of the VOC trio, and it is what makes
a cluster comparison here about the *question* rather than about the data.

**The label convention was measured, not assumed** (19a). The maps are mode
``L`` with values ``0..39`` and ``255``: **255 is void and 0 is a real class**.
88.5% of border pixels are 255 against 13.7% of interior — that is the
depth-projection margin, and a 255 pixel is 2.5x more likely than average to
have depth exactly 0 — while value 0 shows the opposite pattern (2.3% border,
28.6% interior), which is ``wall`` dominating an indoor scene. So it is VOC's
shape exactly: contiguous zero-indexed classes plus a 255 void, which
``load_label_map`` and ``--ignore-index 255`` already handle with no new loader
and no fifth validity convention.
"""

from visbench.registry import register_task
from visbench.tasks.high_level.semantic_segmentation import SemanticSegmentationTask

__all__ = ["SceneParsingTask", "NYU40_CLASSES", "NYU40_VOID"]

#: The NYU40 label set, as this copy stores it: forty classes at 0..39.
NYU40_CLASSES = 40

#: The value an unlabelled pixel carries in these maps. Passed as the loader's
#: ``ignore_index`` so it becomes ``IGNORE_INDEX`` before the loss sees it;
#: leaving it at 255 would train a forty-class head to predict class 255 and
#: score it, which reads as a weak representation rather than a wiring error.
NYU40_VOID = 255


@register_task("scene_parsing")
class SceneParsingTask(SemanticSegmentationTask):
    """Dense category prediction over indoor scenes, forty classes.

    Every mechanical part — the head, the AdamW schedule, the two mIoUs and the
    ``IGNORE_INDEX`` handling that makes loss and metric drop the same pixels —
    is inherited unchanged. Only the identity of the number changes.
    """

    level = "high_level"
    display_name = "Scene parsing"

    def __init__(self, num_classes: int = NYU40_CLASSES, *args, **kwargs) -> None:
        """``num_classes`` defaults here where the base deliberately has none.

        :class:`SemanticSegmentationTask` refuses a default because the count is
        a property of whatever dataset is passed and a wrong one does not raise
        — it trains a head that cannot express some categories and reports a
        plausible number. This probe is **named for a label set**, so the count
        is a protocol pin rather than a caller's business, the way ``corner``
        pins its frame set. It still travels in ``task_params``, so a run at
        another count lands in its own comparability group.
        """
        super().__init__(num_classes, *args, **kwargs)
        # The base hardcodes self.name = "semantic_segmentation"; this is the
        # one field that has to move, and it is what keeps the two boards apart.
        self.name = "scene_parsing"

    def describe(self) -> dict:
        """Task metadata plus a ``protocol`` naming this as the parsing probe.

        The task name already separates the comparability groups, so this string
        changes nothing about ranking — it says in the record what kind of
        number this is, and a test pins it *different* from
        ``visbench_semantic_seg``, since the failure mode two probes on one
        implementation have is collapsing into one board.
        """
        described = super().describe()
        described["task_params"]["protocol"] = "visbench_scene_parsing"
        return described

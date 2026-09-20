"""Vehicle recognition — a linear probe on pooled features, on car-model labels.

Mechanically identical to
:class:`~visbench.tasks.high_level.classification.ClassificationTask`, to
:class:`~visbench.tasks.high_level.scene_classification.SceneClassificationTask`
and to
:class:`~visbench.tasks.high_level.fine_grained_classification.FineGrainedClassificationTask`:
one linear layer, AdamW, top-1/top-5. The fourth question asked on that one
implementation, and the second at *subordinate* granularity.

**What distinguishes it from the bird board, given both are subordinate.** CUB's
two hundred species share a body plan and differ in the colour of a wing bar or
the shape of a beak — evidence that is largely *textural* and spread over the
animal. A car model is a **manufactured rigid object**: two models of the same
class share a silhouette and differ in badge, grille and lamp geometry, at
whatever scale the photograph happens to put them. Whether a representation
keeps one kind of detail says little about whether it keeps the other, and the
measurement agrees — see below.

**Why a distinct probe and not a second dataset for**
``fine_grained_classification``. :func:`~visbench.results.render.board_for`
renders exactly one table per task and refuses a task with more than one
comparability group, and
:func:`~visbench.results.leaderboard.comparability_key` groups by dataset name
and fingerprint. A Cars record under ``task="fine_grained_classification"``
would therefore not join the CUB board — it would make that board
*unrenderable*. That sibling's own docstring anticipates this probe by name, as
"the same probe under a different fingerprint"; a different fingerprint is
precisely what needs a different task name to be rankable at all.

**It earns the board by ranking differently, which was measured before it was
built.** The published board is Spearman **+0.863** against CUB over all
thirteen backbones, spanning **0.8603** to **0.4353** — high, but a dozen of the
corpus's existing board pairs are *more* correlated than that, and the leader
changes: ``siglip_vitb16`` is **first** here and fifth on CUB. Web image-text
pretraining ahead of everything else on manufactured categories is the kind of
statement this corpus exists to make. Had the ordering merely reproduced CUB's,
this probe should not have shipped — the standard relative depth was rejected
against.

**Its closest neighbour is not the other subordinate board.** The strongest
partner is ``scene_classification`` at **+0.901**, ahead of
``fine_grained_classification``'s +0.863 — so what these boards share is not
*granularity*, which is the property this probe was expected to isolate. Mean
rho is +0.525 against the high-level tier, +0.149 against mid-level and
**−0.190** against low-level; read it in ``CORPUS_FINDINGS.md`` rather than
here.

**The split is VisBench's own, and that is not a detail.** The Stanford Cars
copy on this machine is *not* the official 8,144/8,041 split: its train side
holds 8,148 images, eleven of which are the same image filed under two class
directories, its test side has seven more such pairs, and one train image is a
blank white placeholder. ``scripts/stage_cars_split.py`` pins a cleaned
**8,125 / 8,026** split — every exclusion named in
``data/cars_split_manifest.json``, no image appearing twice under any label —
and the probe's docs page states, as this docstring does, that **numbers
measured on it are not comparable with published Stanford Cars results.** A
cross-class duplicate cannot be repaired by choosing a label, so both copies
go; what remains is reproducible from the manifest by anyone holding the same
raw copy.
"""

from visbench.registry import register_task
from visbench.tasks.high_level.classification import ClassificationTask

__all__ = ["VehicleClassificationTask"]


@register_task("vehicle_classification")
class VehicleClassificationTask(ClassificationTask):
    """Linear-probe classification of car models — subordinate, manufactured.

    Every mechanical part is inherited unchanged from
    :class:`ClassificationTask`: the lazily-built ``nn.Linear``, the AdamW
    schedule, the optional standardiser and its ``probe_state`` serialisation.
    Only the identity of the number changes, which is the whole design of this
    family — four probes, one implementation, four questions.

    The schedule defaults are the parent's, checked rather than assumed on the
    same grounds the CUB board was: 196 classes over ~8k training images is the
    case most likely to underfit, and it does not. On the published board
    ``train_top1`` reaches **1.0000 on seven of thirteen rows and 0.985 or
    better on the rest** — near-interpolating everywhere, so the gap to the
    validation score is *generalisation* rather than an unconverged probe, which
    is the condition under which a weak number is a property of the
    representation. CUB reaches a flat 1.0000 where this board does not, which
    is the one place the two differ mechanically and the reason to read
    ``train_top1`` here before quoting a low row. The board spans **0.8603**
    (``siglip_vitb16``) to **0.4353** (``mae_vitb16``).

    One caveat belongs with every number here, in the same place its sibling
    keeps one: ImageNet-1k contains several car classes ("sports car",
    "convertible", "limousine"), but none at *model* granularity, so the
    in-distribution-recall confound that
    ``CORPUS_FINDINGS.md`` records for the Imagenette board is weaker here than
    on CUB rather than absent. The measurement is consistent with that —
    ``supervised_vitb16`` is **twelfth of thirteen** here against eleventh on
    CUB — but one board cannot separate "the confound is weak" from "this
    backbone is weak at this task", and nothing here claims it does.
    """

    level = "high_level"
    display_name = "Vehicle recognition"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The base hardcodes self.name = "classification"; this is the one field
        # that has to move, and it is what keeps the four boards apart.
        self.name = "vehicle_classification"

    def describe(self) -> dict:
        """Task metadata plus a ``protocol`` naming this as the vehicle probe.

        Object classification records carry no ``protocol`` key; the other three
        questions on this implementation each carry their own, purely for
        provenance. The task name already separates the comparability groups, so
        the string changes nothing about ranking — it says in the record what
        kind of number this is, and a test pins all four apart because the
        failure mode is two of them collapsing into one board.
        """
        described = super().describe()
        described["task_params"]["protocol"] = "visbench_vehicle_linear_probe"
        return described

"""Surface normal estimation — v0.2.

The protocol comes from probe3d (El Banani et al., CVPR 2024,
arXiv:2404.08476), whose ``evals/utils/losses.py`` and ``evals/utils/metrics.py``
are MIT licensed. This reproduces its ``configs/probe/snorm_dpt.yaml`` plus
``configs/optimizer/ten_epoch.yaml``:

===============  ==========================================================
prediction       3 channels (a direction), L2-normalised
loss             angular error, uncertainty-aware (Bae et al.) by default
optimiser        AdamW, lr 5e-4, 10 epochs, 1.5 warmup, cosine decay
backbone         frozen (their ``model_lr: 0``)
===============  ==========================================================

**Uncertainty-aware by default**, matching probe3d's config. The head emits a
fourth channel, a concentration parameter kappa, and the loss lets it discount
pixels it cannot predict — object boundaries and thin structures, mostly —
instead of being dragged around by them. It costs one extra output channel and
changes no metric: scoring reads the first three channels either way. Pass
``uncertainty_aware=False`` for the plain angular loss.

That default has a failure mode worth knowing about before the first run. Near
chance accuracy kappa settles low enough to all but switch the direction's
supervision off, and weak supervision keeps accuracy near chance — a probe that
falls in reports a chance-level score with no error and no obvious symptom.
:meth:`SurfaceNormalTask.fit` gives the measured dynamics, detects it, and
warns. It is inherited from probe3d rather than introduced here, and is left in
place because matching their numbers is the only reason to have borrowed the
protocol.

**Where the ground truth comes from matters.** NYU surface normals are
conventionally *derived* — GeoNet's or Ladicky's extraction from the depth
maps, not a sensor product — and the three sources disagree enough to move the
numbers. :meth:`describe` records ``normal_source`` verbatim so a result says
which was used; it defaults to ``None``, and a comparison across runs that
leave it unset is a comparison of unknown things.

**Validity.** probe3d masks with ``depth > 0``, because its NYU loader carries
the depth map alongside. A normal map on its own carries the same information
as a zero-length vector, which is what
:func:`~visbench.metrics.dense.surface_normal_metrics` and the loss here use.
"""

import warnings
from typing import Any

import torch

from visbench.metrics.dense import surface_normal_metrics
from visbench.registry import register_task
from visbench.tasks.dense_base import DenseTrainingTask
from visbench.types import MetricsDict

__all__ = ["SurfaceNormalTask", "angular_loss"]


def angular_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    uncertainty_aware: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """probe3d's ``angular_loss``: mean angular error over valid pixels.

    The cosine is clamped to ``[-1 + eps, 1 - eps]`` before ``acos``, not to
    ``[-1, 1]`` as the metric does. The tighter bound is deliberate: ``acos``
    has infinite derivative at both ends, so a prediction that lands exactly on
    a target would produce an infinite gradient and destroy the head in one
    step. The metric has no gradient and can afford the exact bound.

    Parameters
    ----------
    pred:
        ``(B, 3, H, W)``, or ``(B, 4, H, W)`` when ``uncertainty_aware`` — the
        fourth channel is the raw concentration, passed through ``elu`` here.
    mask:
        ``(B, H, W)`` bool or float. Defaults to where ``target`` has non-zero
        length, matching the metric's convention.

    Notes
    -----
    Under ``uncertainty_aware`` the loss is Bae et al.'s negative log
    likelihood for an angular von Mises-Fisher-like distribution: the head
    predicts kappa alongside the direction, is rewarded for being confident
    where it is right, and pays ``kappa_reg`` for confidence everywhere else.
    A pixel it genuinely cannot call — an object boundary, where the true
    normal flips within one patch — can be conceded cheaply rather than
    dominating the gradient.

    Unlike the reference this returns a zero *connected to the graph* when no
    pixel is valid, rather than the NaN that ``.mean()`` of an empty selection
    produces. probe3d drops into a debugger there; a benchmark library cannot.
    """
    if pred.ndim != 4 or target.ndim != 4 or target.shape[1] != 3:
        raise ValueError(
            f"Expected (B, 3, H, W) targets and (B, C, H, W) predictions, got "
            f"{tuple(target.shape)} and {tuple(pred.shape)}"
        )
    wanted = 4 if uncertainty_aware else 3
    if pred.shape[1] != wanted:
        raise ValueError(
            f"uncertainty_aware={uncertainty_aware} needs {wanted} predicted channels, "
            f"got {pred.shape[1]}"
        )

    valid = (target.norm(dim=1) > 0) if mask is None else mask.bool()
    if not valid.any():
        # Keeps the graph connected: a bare zero would detach the head and
        # silently skip this batch's gradient.
        return pred.sum() * 0.0

    cosine = torch.cosine_similarity(pred[:, :3], target, dim=1)
    error = cosine.clamp(min=-1 + eps, max=1 - eps).acos()

    if not uncertainty_aware:
        return error[valid].mean()

    # elu + 1.01 keeps kappa positive with a floor of 0.01, as in the paper —
    # a kappa that could reach zero would make the likelihood term vanish and
    # let the head opt out of predicting anything at all.
    kappa = torch.nn.functional.elu(pred[:, 3]) + 1.01
    regulariser = torch.log1p((-kappa * torch.pi).exp()) - (kappa.pow(2) + 1).log()
    return (regulariser + kappa * error)[valid].mean()


@register_task("surface_normal")
class SurfaceNormalTask(DenseTrainingTask):
    """Pixel-wise 3D surface orientation, via a pluggable head.

    As with depth, ``head="linear"`` is the number that compares
    *representations* and ``head="dpt"`` is probe3d's own choice; report both,
    or say which.
    """

    level = "mid_level"
    display_name = "Surface normal estimation"
    target_noun = "target normal maps"
    target_channels = 3

    def __init__(
        self,
        head: str = "linear",
        layers: list[int] | None = None,
        uncertainty_aware: bool = True,
        hidden_dim: int = 512,
        epochs: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        warmup_epochs: float = 1.5,
        normal_source: str | None = None,
        head_kwargs: dict | None = None,
        device: str | None = None,
    ) -> None:
        """Configure the probe; the head is built lazily in :meth:`fit`.

        Parameters
        ----------
        head, layers, epochs, lr, warmup_epochs:
            See :class:`~visbench.tasks.dense_base.DenseTrainingTask`.
        uncertainty_aware:
            Predict a fourth confidence channel and use Bae et al.'s loss.
            probe3d's default, and this one.
        normal_source:
            Free-text provenance for the ground truth — ``"geonet"``,
            ``"ladicky"``, ``"sensor"``, a URL. Recorded, never interpreted.
            See the module docstring for why it is worth setting.
        """
        super().__init__(
            head=head,
            layers=layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            head_kwargs=head_kwargs,
            device=device,
        )
        self.name = "surface_normal"
        self.uncertainty_aware = uncertainty_aware
        self.normal_source = normal_source

        #: Fraction of pixels whose kappa sat on its floor during the final
        #: training epoch — see :meth:`fit`. A diagnostic, not a result.
        self.kappa_at_floor: float | None = None
        self._floor_count = 0
        self._kappa_count = 0

    # -- kappa collapse ------------------------------------------------------

    #: Below this, kappa's ``elu`` has saturated and its gradient is dead.
    #: ``elu(x) + 1.01 < 0.02`` needs ``x < -4.6``, which no live unit reaches.
    _kappa_floor_threshold = 0.02

    #: Warn once more than this fraction of pixels have collapsed.
    _kappa_collapse_warn = 0.5

    def fit(self, features: Any, labels: Any | None = None) -> "SurfaceNormalTask":
        """Train the head, then check whether the uncertainty channel died.

        The uncertainty-aware loss has a silent failure mode at chance level.
        Kappa settles where the angular term and the regulariser balance, which
        makes it a smooth function of how well the head is doing:

        ===================  ==========
        angular error        kappa
        ===================  ==========
        30 deg               3.5
        60 deg               1.2
        80 deg               0.38
        90 deg (chance)      0.05
        ===================  ==========

        That is the intended behaviour — until the head is near chance, where
        kappa lands around 0.05 and scales the direction's gradient by 1/20.
        Weak supervision keeps the head near chance, which keeps kappa low: the
        two hold each other down. With a real head the pre-activation behind
        kappa can saturate the ``elu`` outright, pinning it at the 0.01 floor
        where its own gradient is ``exp(-4.6)`` and recovery is effectively
        over. A DINOv2 linear probe on a twelve-image split does exactly this,
        and reports a chance-level score with no error and no warning.

        Whether a given run falls in depends on head initialisation, so this is
        detected per run rather than predicted. It is inherited from probe3d,
        not introduced here, and it does not bite on NYUv2, where the head is
        well under 64 degrees within the warmup. Left in place because silently
        switching to the plain loss would make VisBench's numbers incomparable
        with the published ones — the only reason to have borrowed the protocol.

        If it fires: pass ``uncertainty_aware=False``, or give the direction a
        head start with a longer warmup or a higher learning rate.
        """
        super().fit(features, labels)
        if self.uncertainty_aware and self._kappa_count:
            self.kappa_at_floor = self._floor_count / self._kappa_count
            if self.kappa_at_floor > self._kappa_collapse_warn:
                warnings.warn(
                    f"{self.kappa_at_floor:.0%} of predicted kappa values collapsed to their "
                    "floor during the last epoch, so the angular term was scaled by ~1/100 "
                    "and the direction was barely supervised. Any score from this probe is "
                    "close to meaningless — expect roughly chance. Re-run with "
                    "uncertainty_aware=False, or give the direction a head start with a "
                    "longer warmup or a higher lr. See SurfaceNormalTask.fit for the mechanism.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self

    def _on_epoch_start(self) -> None:
        """Count kappa collapse over the final epoch, not over all of training.

        Kappa is *expected* to be low early on, while the direction is still
        random; only its state at the end says whether the channel is dead.
        """
        self._floor_count = 0
        self._kappa_count = 0

    @property
    def out_channels(self) -> int:
        """Three direction channels, plus one for kappa when uncertainty-aware."""
        return 4 if self.uncertainty_aware else 3

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        """L2-normalise the direction; leave any kappa channel alone.

        The loss and the metric both compare by *cosine*, so normalising
        changes neither — it is done so that :meth:`predict` hands back an
        actual unit normal rather than an unscaled direction whose length is
        whatever the last linear layer happened to produce.

        ``eps`` guards the zero vector: an untrained head predicts near-zero
        almost everywhere on the first step, and dividing by that length would
        put NaN into the loss before a single gradient had been taken.
        """
        direction = torch.nn.functional.normalize(raw[:, :3], dim=1, eps=1e-8)
        if raw.shape[1] == 3:
            return direction
        return torch.cat([direction, raw[:, 3:]], dim=1)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.uncertainty_aware:
            with torch.no_grad():
                kappa = torch.nn.functional.elu(pred[:, 3]) + 1.01
                self._floor_count += int((kappa < self._kappa_floor_threshold).sum())
                self._kappa_count += kappa.numel()
        return angular_loss(pred, target, uncertainty_aware=self.uncertainty_aware)

    def _batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> MetricsDict:
        """``{"d1", "d2", "d3", "rmse", "mean", "median"}`` per probe3d."""
        return surface_normal_metrics(pred, target)

    def _task_params(self) -> dict:
        return {
            "uncertainty_aware": self.uncertainty_aware,
            "normal_source": self.normal_source,
        }

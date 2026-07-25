"""Task implementations, grouped by level.

Levels follow Chen, Marks & Cheng (arXiv:2411.17474):

* ``high_level`` — semantic / category understanding.
* ``mid_level``  — geometry and generic structure prior to semantic labeling.
  This is the paper's core contribution area and where VisBench aims to be
  strongest relative to existing tooling.
* ``low_level``  — placeholder only until v0.3+.

Importing this package runs the ``@register_task`` decorators.
"""

from visbench.tasks.base import BaseTask
from visbench.tasks.dense_base import DenseTrainingTask

__all__ = ["BaseTask", "DenseTrainingTask"]

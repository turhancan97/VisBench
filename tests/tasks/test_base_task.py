"""Contract tests for BaseTask — filled in at build step 3.

* ``evaluate`` returns a flat dict of floats — no nesting, no tensors, since
  the result schema depends on it.
* ``fit`` is a no-op returning ``self`` for zero-shot tasks.
* A task requests pooling from the backbone; the backbone never chooses.
* ``evaluate`` prints nothing — the caller writes the structured record.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Scaffold only — implemented at build step 3.")


def test_evaluate_returns_flat_float_dict():
    raise NotImplementedError


def test_zero_shot_fit_is_noop():
    raise NotImplementedError


def test_task_drives_pooling_choice():
    raise NotImplementedError


def test_evaluate_does_not_print():
    raise NotImplementedError

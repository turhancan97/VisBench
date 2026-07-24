"""Contract tests for BaseBackbone — filled in at build step 2.

The contract these will enforce, once DINOv2 is implemented:

* ``extract_features`` returns all three keys, with ``dense`` ``(B, C, H, W)``,
  ``pooled`` ``(B, C)`` and ``grid_hw`` matching ``dense.shape[-2:]``.
* ``pooling="default"`` resolves to CLS when ``has_cls_token``, else mean.
* ``pooling="cls"`` on a backbone without a CLS token raises rather than
  silently falling back.
* Parameters are frozen and the module is in eval mode after construction.
* ``layers`` with more than one entry raises until v0.2.

Written as a checklist now so step 2 has a definition of done.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Scaffold only — implemented at build step 2.")


def test_extract_features_returns_dense_and_pooled():
    raise NotImplementedError


def test_default_pooling_follows_architecture():
    raise NotImplementedError


def test_cls_pooling_without_cls_token_raises():
    raise NotImplementedError


def test_backbone_is_frozen_and_eval():
    raise NotImplementedError


def test_multilayer_request_raises_in_v01():
    raise NotImplementedError

"""Contract tests for the feature cache — filled in at build step 2.

The cache is the piece most likely to fail silently, so these matter more than
their line count suggests:

* A second extraction of the same image hits the cache; the backbone forward
  runs exactly once (assert via a call-counting fake backbone).
* Changing any key component — image, backbone key, layer, pooling — misses.
* Two identical images under different filenames share one entry.
* ``enabled=False`` always misses without changing call sites.
* An interrupted write leaves no readable entry (atomic writes).
"""

import pytest

pytestmark = pytest.mark.skip(reason="Scaffold only — implemented at build step 2.")


def test_second_extraction_hits_cache():
    raise NotImplementedError


def test_key_components_all_affect_identity():
    raise NotImplementedError


def test_identical_images_share_entry():
    raise NotImplementedError


def test_disabled_cache_always_misses():
    raise NotImplementedError


def test_partial_write_is_not_a_hit():
    raise NotImplementedError

"""The QuickGELU guard, tested without downloading a checkpoint.

``tests/backbones/test_clip.py`` covers this too, but that whole module is
``slow`` because it builds a real CLIP — and CI does not run ``-m slow``. The
guard was broken for its entire life and the only test that could have caught
it never ran. So the detection logic lives in a helper that takes a list of
warnings, and is exercised here in the fast suite against open_clip's *verbatim*
wording.

Nothing here imports open_clip: CI installs ``.[dev]``, not ``.[clip]``.
"""

import warnings

import pytest

from visbench.backbones.clip import _promote_quickgelu_warning

# Copied verbatim from open_clip 2.32.0, factory.py:388 and :393 — the two
# directions it warns in. Hardcoded rather than imported so this suite runs
# without the clip extra; test_wording_canary below catches upstream drift.
WEIGHTS_HAVE_CONFIG_LACKS = (
    "These pretrained weights were trained with QuickGELU activation but the "
    'model config does not have that enabled. Consider using a model config with a "-quickgelu" '
    "suffix or enable with a flag."
)
CONFIG_HAS_WEIGHTS_LACK = (
    "The pretrained weights were not trained with QuickGELU but this activation "
    "is enabled in the model config, consider using a model config without QuickGELU or disable "
    "override flags."
)


def record(*messages, category=UserWarning):
    """Return WarningMessage objects as ``catch_warnings(record=True)`` would."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for message in messages:
            warnings.warn(message, category, stacklevel=2)
    return caught


class TestItRaises:
    def test_weights_have_quickgelu_but_config_does_not(self):
        with pytest.raises(RuntimeError, match="QuickGELU mismatch"):
            _promote_quickgelu_warning(record(WEIGHTS_HAVE_CONFIG_LACKS), "ViT-B-16", "openai")

    def test_the_other_direction_too(self):
        """open_clip warns both ways and both are wrong activations."""
        with pytest.raises(RuntimeError, match="QuickGELU mismatch"):
            _promote_quickgelu_warning(
                record(CONFIG_HAS_WEIGHTS_LACK), "ViT-B-16-quickgelu", "laion2b_s34b_b88k"
            )

    def test_the_error_names_the_pairing_that_caused_it(self):
        with pytest.raises(RuntimeError) as excinfo:
            _promote_quickgelu_warning(record(WEIGHTS_HAVE_CONFIG_LACKS), "ViT-B-16", "openai")
        assert "ViT-B-16" in str(excinfo.value)
        assert "openai" in str(excinfo.value)

    def test_it_quotes_open_clips_own_text(self):
        """The reader needs the upstream wording, not just our summary of it."""
        with pytest.raises(RuntimeError, match="pretrained weights"):
            _promote_quickgelu_warning(record(WEIGHTS_HAVE_CONFIG_LACKS), "ViT-B-16", "openai")

    def test_matching_is_case_insensitive(self):
        with pytest.raises(RuntimeError, match="QuickGELU mismatch"):
            _promote_quickgelu_warning(record("quickgelu activation differs"), "m", "p")

    def test_a_mismatch_after_an_unrelated_warning_still_raises(self):
        with pytest.raises(RuntimeError, match="QuickGELU mismatch"):
            _promote_quickgelu_warning(
                record("something unrelated", WEIGHTS_HAVE_CONFIG_LACKS), "m", "p"
            )


class TestItStaysOutOfTheWay:
    def test_no_warnings_is_not_an_error(self):
        _promote_quickgelu_warning([], "ViT-B-16-quickgelu", "openai")

    def test_an_unrelated_warning_is_not_an_error(self):
        _promote_quickgelu_warning(record("torch deprecated something"), "m", "p")

    def test_unrelated_warnings_are_re_emitted_not_swallowed(self):
        """Recording suppresses warnings; this guard must not hide other ones."""
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            _promote_quickgelu_warning(record("torch deprecated something"), "m", "p")

        assert len(seen) == 1
        assert "torch deprecated something" in str(seen[0].message)

    def test_re_emission_preserves_the_category(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            _promote_quickgelu_warning(record("going away", category=DeprecationWarning), "m", "p")

        assert seen[0].category is DeprecationWarning


class TestRegression:
    def test_the_original_filter_could_never_have_matched(self):
        """The guard used to look for a phrase open_clip does not emit."""
        assert "quickgelu mismatch" not in WEIGHTS_HAVE_CONFIG_LACKS.lower()
        assert "quickgelu mismatch" not in CONFIG_HAS_WEIGHTS_LACK.lower()

    def test_wording_canary(self):
        """Fail loudly here if open_clip stops saying 'QuickGELU' at all.

        Skipped without the clip extra, which is how CI runs. Local dev has it,
        and that is where a dependency bump gets noticed.
        """
        factory = pytest.importorskip("open_clip.factory")
        source = __import__("inspect").getsource(factory)
        assert "QuickGELU" in source, (
            "open_clip no longer mentions QuickGELU; _QUICKGELU_MARKER needs revisiting"
        )

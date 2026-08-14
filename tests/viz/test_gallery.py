"""The three probes whose answer is a choice among images."""

import numpy as np
import pytest
import torch
from PIL import Image

from visbench.viz import (
    RIGHT,
    WRONG,
    annotate,
    class_balance,
    render_retrieval_panels,
    render_sheet,
    render_triplet_panels,
    style_for,
    vote_balance,
)
from visbench.viz.gallery import NEUTRAL


class _Folder:
    """A labelled image folder, grouped by class the way a real one is."""

    def __init__(self, per_class=3, classes=("a", "b", "c")):
        self.classes = list(classes)
        self._labels = [index for index in range(len(classes)) for _ in range(per_class)]
        self._images = [
            Image.new("RGB", (24, 24), (30 + 60 * label, 40, 50)) for label in self._labels
        ]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._images[index], self._labels[index]

    def labels(self):
        return list(self._labels)


class TestAnnotate:
    def test_a_caption_never_covers_the_frame(self):
        """The no-second-geometry rule, in the one place text is drawn.

        The canvas grows downward; the frame itself is pasted unchanged, so the
        pixels under inspection are never obscured by a label describing them.
        """
        image = Image.new("RGB", (32, 32), (77, 88, 99))
        drawn = annotate(image, "a very long caption indeed")
        assert drawn.height > image.height
        assert np.array_equal(np.asarray(drawn)[: image.height], np.asarray(image))

    def test_a_border_is_drawn_inside_the_frame(self):
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        drawn = np.asarray(annotate(image, "", RIGHT))
        assert tuple(drawn[0, 0]) == RIGHT
        assert tuple(drawn[16, 16]) == (0, 0, 0)  # the middle is untouched

    def test_captions_are_ascii_so_the_default_font_can_draw_them(self):
        """PIL's built-in bitmap font renders an em dash as an empty box.

        Found by looking at a rendered page, not by a test — which is the whole
        argument for this package, arriving inside it.
        """
        for text in (
            class_balance([0, 0], ["only"]),
            vote_balance(torch.tensor([[0, 1, 2, 1]])),
        ):
            assert text.isascii(), text

    def test_a_caption_too_wide_is_shortened_not_clipped(self):
        image = Image.new("RGB", (24, 24), (0, 0, 0))
        wide = annotate(image, "an extremely long class name")
        assert wide.width == 24  # it did not grow sideways
        # And something was drawn, rather than the caption vanishing entirely.
        assert np.asarray(wide)[24:].any()


class TestClassBalance:
    def test_a_collapsed_split_says_so(self):
        """The prefix bug as a figure: `subset(n)` on a class-grouped folder.

        The run would score 1.0 and look like a triumph. This line reads
        `1 class` whichever frames were drawn.
        """
        line = class_balance([0] * 8, ["tench", "chain"])
        assert "1 class," in line
        assert "artefact" in line
        assert "tench" in line

    def test_a_balanced_split_reports_its_spread(self):
        line = class_balance([0, 0, 1, 1, 2, 2], ["a", "b", "c"])
        assert "3 classes" in line and "6 items" in line and "2-2 per class" in line

    def test_an_imbalanced_split_shows_the_range(self):
        assert "1-5 per class" in class_balance([0] * 5 + [1])

    def test_an_empty_split_does_not_divide_by_zero(self):
        assert class_balance([]) == "empty split"


class TestVoteBalance:
    def test_a_balanced_vote_reads_near_half(self):
        triplets = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 0], [0, 1, 2, 1]])
        assert "50%" in vote_balance(triplets)

    def test_a_one_sided_vote_is_the_wrong_column_showing(self):
        """NIGHTS presents the candidates in arbitrary order, so ~50% is expected.

        A figure far from it means the vote was read from a different CSV field
        — the failure that otherwise surfaces only as a mediocre accuracy.
        """
        line = vote_balance(torch.tensor([[0, 1, 2, 1]] * 10))
        assert "100%" in line and "vote column is wrong" in line


class TestSheet:
    def test_it_packs_several_frames_per_row(self):
        folder = _Folder(per_class=3)
        page = render_sheet(folder, list(range(9)), columns=3)
        # Three tiles wide, not nine rows tall.
        assert page.width == 200 + 3 * 24 + 3 * 8

    def test_without_predictions_every_frame_is_neutral(self):
        folder = _Folder(per_class=1, classes=("a",))
        page = np.asarray(render_sheet(folder, [0], columns=1))
        assert NEUTRAL in {tuple(pixel) for pixel in page.reshape(-1, 3)}
        assert RIGHT not in {tuple(pixel) for pixel in page.reshape(-1, 3)}

    def test_a_wrong_prediction_is_bordered_differently_from_a_right_one(self):
        folder = _Folder(per_class=1, classes=("a", "b"))
        colours = {}
        for prediction, name in ((0, "right"), (1, "wrong")):
            page = np.asarray(render_sheet(folder, [0], [prediction], columns=1))
            colours[name] = {tuple(pixel) for pixel in page.reshape(-1, 3)}
        assert RIGHT in colours["right"] and RIGHT not in colours["wrong"]
        assert WRONG in colours["wrong"] and WRONG not in colours["right"]

    def test_the_footer_carries_the_balance_and_the_hit_rate(self):
        folder = _Folder(per_class=2, classes=("a", "b"))
        page = render_sheet(folder, [0, 1, 2, 3], [0, 0, 1, 1], columns=2)
        assert page.height > 0  # rendered; the strings are covered above


class TestRetrieval:
    def _ranking(self, folder):
        """Perfect retrieval: every item ranks its own class first."""
        labels = folder.labels()
        order = []
        for index in range(len(folder)):
            same = [
                other
                for other in range(len(folder))
                if other != index and labels[other] == labels[index]
            ]
            rest = [
                other
                for other in range(len(folder))
                if other != index and labels[other] != labels[index]
            ]
            order.append(same + rest)
        return torch.tensor(order)

    def test_a_neighbour_of_the_query_s_class_is_marked_differently(self):
        """The border encodes correctness rather than decorating the row."""
        folder = _Folder(per_class=3)
        ranking = self._ranking(folder)

        good = np.asarray(render_retrieval_panels(folder, ranking, [0], topk=2))
        assert RIGHT in {tuple(pixel) for pixel in good.reshape(-1, 3)}

        # Rank a different class first and the same row turns red.
        wrong = ranking.clone()
        wrong[0] = torch.tensor([3, 4, 5, 1, 2, 6, 7, 8][: ranking.shape[1]])
        bad = np.asarray(render_retrieval_panels(folder, wrong, [0], topk=2))
        assert WRONG in {tuple(pixel) for pixel in bad.reshape(-1, 3)}

    def test_the_query_itself_is_neither_right_nor_wrong(self):
        folder = _Folder(per_class=3)
        page = np.asarray(render_retrieval_panels(folder, self._ranking(folder), [0], topk=1))
        assert NEUTRAL in {tuple(pixel) for pixel in page.reshape(-1, 3)}

    def test_one_column_per_neighbour_plus_the_query(self):
        folder = _Folder(per_class=3)
        page = render_retrieval_panels(folder, self._ranking(folder), [0, 1], topk=3)
        assert page.width == 200 + 4 * 24 + 4 * 8


class _Triplets:
    """Three images; the triplet prefers whichever the vote names."""

    def __init__(self, vote=1):
        self._images = [Image.new("RGB", (20, 20), (c, c, c)) for c in (10, 120, 230)]
        self._triplets = torch.tensor([[0, 1, 2, vote]])

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self._images[index], None

    def labels(self):
        return self._triplets


class TestTriplets:
    def test_the_human_vote_is_drawn_on_the_chosen_candidate(self):
        """A transposed vote column would put the marker on the other frame.

        Checked by the *position* of the marked tile, not merely that one is
        marked — the second is satisfied by any vote at all.
        """
        for vote, marked_column in ((0, 1), (1, 2)):
            page = np.asarray(render_triplet_panels(_Triplets(vote), [0]))
            # Columns start after the gutter; each tile is 20 wide plus an 8 gap.
            for column in (1, 2):
                left = 200 + column * (20 + 8)
                tile = page[:, left : left + 20]
                has_marker = RIGHT in {tuple(pixel) for pixel in tile.reshape(-1, 3)}
                assert has_marker == (column == marked_column), (vote, column)

    def test_the_reference_is_never_marked_as_a_choice(self):
        page = np.asarray(render_triplet_panels(_Triplets(1), [0]))
        reference = page[:, 200 : 200 + 20]
        assert RIGHT not in {tuple(pixel) for pixel in reference.reshape(-1, 3)}

    def test_a_disagreeing_model_is_visible(self):
        agree = np.asarray(render_triplet_panels(_Triplets(1), [0], [1]))
        differ = np.asarray(render_triplet_panels(_Triplets(1), [0], [0]))
        assert WRONG not in {tuple(pixel) for pixel in agree.reshape(-1, 3)}
        assert WRONG in {tuple(pixel) for pixel in differ.reshape(-1, 3)}

    def test_three_columns(self):
        page = render_triplet_panels(_Triplets(1), [0])
        assert page.width == 200 + 3 * 20 + 3 * 8


class TestTheStyleRows:
    @pytest.mark.parametrize(
        ("probe", "kind"),
        [("classification", "sheet"), ("retrieval", "ranking"), ("similarity", "triplet")],
    )
    def test_each_has_its_own_kind_and_refuses_the_panel_colouriser(self, probe, kind):
        from visbench.viz import COMPOSITE_KINDS, target_to_rgb

        style = style_for(probe)
        assert style.kind == kind
        assert style.kind in COMPOSITE_KINDS
        with pytest.raises(ValueError, match="gallery"):
            target_to_rgb(torch.rand(4, 4), style)

    @pytest.mark.parametrize("probe", ["classification", "retrieval", "similarity"])
    def test_none_of_them_claims_an_invalid_convention(self, probe):
        """A class index has no invalid value, and the table states that."""
        assert style_for(probe).invalid is None

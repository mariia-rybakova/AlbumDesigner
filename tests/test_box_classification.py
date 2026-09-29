"""Square boxes are square by ratio, whatever their size.

49995684's design 1997004 has twelve 0.113 x 0.175 boxes on a 1.962:1 album --
1.26:1 each, nearer a 3:2 landscape photo than a square. The old test,
``|w * album_ar - h| <= 0.05`` in page units, let a box that small be up to 29%
off square, so all twelve were `small square`, portraits filled them freely,
and each kept 53% of its height.

    python -m pytest tests/test_box_classification.py -v
"""

import pytest

from utils.layouts_tools import SQUARE_TOLERANCE, classify_box

ALBUM_AR = 1.9620315074640127


def box(width, height):
    return {'width': width, 'height': height}


def kind(b):
    return classify_box(b, SQUARE_TOLERANCE, ALBUM_AR)[0]


def test_a_small_wide_box_is_landscape_not_square():
    assert kind(box(0.113, 0.175)) == 'landscape'


def test_the_same_shape_is_the_same_class_at_any_size():
    small, large = box(0.113, 0.175), box(0.113 * 4, 0.175 * 4)

    assert kind(small) == kind(large) == 'landscape'


@pytest.mark.parametrize("ratio", [0.94, 0.95, 1.0, 1.05, 1.08])
def test_a_near_square_stays_square(ratio):
    """The design carries 78 boxes at 0.95:1, plainly meant as squares."""
    height = 0.3
    assert 'square' in kind(box(ratio * height / ALBUM_AR, height))


def test_a_full_page_box_is_still_square():
    """The opening and closing pages: half the spread, 0.98:1."""
    assert 'square' in kind(box(0.5, 1.0))


@pytest.mark.parametrize("ratio,expected", [(1.23, 'landscape'), (1.37, 'landscape'),
                                            (0.8, 'portrait'), (0.66, 'portrait')])
def test_clear_shapes_keep_their_orientation(ratio, expected):
    height = 0.2
    assert kind(box(ratio * height / ALBUM_AR, height)) == expected

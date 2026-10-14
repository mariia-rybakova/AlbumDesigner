"""fold.pdf exists only when there is something to show.

An album whose every spread holds more than one photo has nothing a box can
cross the fold on, and a PDF of no pages is worse than no PDF.

    python -m pytest tests/test_fold_visualizer.py -v
"""

import json
import os
import types

import pytest

from stages_visualizer import fold as fold_visualizer
from utils.configs import CONFIGS
from utils.stages_recorder import fold as fold_recorder


def face(x1, x2, blur=300.0):
    return types.SimpleNamespace(bbox=types.SimpleNamespace(x1=x1, y1=0.2, x2=x2, y2=0.4),
                                 blurLevel=blur)


class _Row(dict):
    """A stand-in for the images_df row `assembly_output` hands the recorder."""


FULL_SPREAD = {'x': 0.0, 'y': 0.0, 'width': 1.0, 'height': 1.0, 'orientation': 'landscape'}
PLACEMENT = {'photoId': 123, 'cropX': 0.0, 'cropY': 0.1, 'cropWidth': 1.0, 'cropHeight': 0.8}


@pytest.fixture(autouse=True)
def _enabled(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIGS['save_files'], 'fold', True)
    monkeypatch.setattr(fold_recorder, '_FOLD_OUT_DIR', str(tmp_path / 'fold'))
    fold_recorder.reset()
    yield
    fold_recorder.reset()


def test_an_album_with_no_one_photo_spread_writes_nothing():
    """`visualization.py` skips the stage when the file is absent."""
    assert fold_recorder.flush(project_id=1) is None


def test_a_one_photo_spread_is_recorded_with_its_margins():
    row = _Row(faces_info=[face(0.45, 0.55)], bodies_info=[])
    fold_recorder.record_spread(3, 3444, 3, FULL_SPREAD, PLACEMENT, row, album_ar=2.0)

    path = fold_recorder.flush(project_id=1)
    assert path is not None
    page = json.load(open(path, encoding='utf-8'))['pages'][0]

    assert page['crosses_fold'] and page['fold_u'] == pytest.approx(0.5)
    span = page['spans'][0]
    assert span['kind'] == 'face'
    assert span['limit_x1'] < span['x1'] and span['limit_x2'] > span['x2'], "margin wraps the face"


def test_a_head_estimated_from_a_body_is_marked_as_such():
    row = _Row(faces_info=[], bodies_info=[face(0.40, 0.60, blur=0.0)])
    fold_recorder.record_spread(3, 3444, 3, FULL_SPREAD, PLACEMENT, row, album_ar=2.0)

    page = json.load(open(fold_recorder.flush(1), encoding='utf-8'))['pages'][0]

    assert [s['kind'] for s in page['spans']] == ['body']
    assert page['spans'][0]['weight'] < 1.0


def test_recording_never_raises_on_a_malformed_row():
    """A debug dump must not take the album down with it."""
    fold_recorder.record_spread(3, 3444, 3, {}, {}, None, album_ar=2.0)

    assert fold_recorder.flush(project_id=1) is None


def test_the_renderer_declines_an_empty_page_list(tmp_path, capsys):
    source = tmp_path / 'fold.json'
    source.write_text(json.dumps({'project_id': 1, 'pages': []}), encoding='utf-8')
    out = tmp_path / 'fold.pdf'

    fold_visualizer.render(str(source), str(tmp_path), str(out))

    assert not os.path.exists(out)
    assert 'no one-photo spreads' in capsys.readouterr().out


def test_an_image_column_maps_through_the_crop_onto_the_page():
    """The fold at image 0.5 of a full-spread box lands mid-page."""
    page = {'crop': {'x': 0.0, 'width': 1.0}, 'box': {'x': 0.0, 'width': 1.0}}

    assert fold_visualizer._image_x_to_page(0.5, page, 1000.0) == pytest.approx(500.0)


def test_a_box_pushed_right_moves_its_columns_with_it():
    page = {'crop': {'x': 0.0, 'width': 1.0}, 'box': {'x': 0.30, 'width': 0.669}}

    assert fold_visualizer._image_x_to_page(0.0, page, 1000.0) == pytest.approx(300.0)
    assert fold_visualizer._image_x_to_page(1.0, page, 1000.0) == pytest.approx(969.0)

"""Tests for the predefined-layout input contract.

Runnable with pytest, or directly: ``python -m src.predefined.tests.test_models``.
"""
from src.predefined.models import PredefinedLayoutInput, PredefinedSpread


def test_absent_block_returns_none():
    assert PredefinedLayoutInput.from_request({}) is None
    assert PredefinedLayoutInput.from_request({"predefinedLayout": None}) is None
    assert PredefinedLayoutInput.from_request({"predefinedLayout": {}}) is None


def test_parses_spreads_photos_only():
    content = {"predefinedLayout": {"spreads": [
        {"photoIds": ["a", "b", "c"]},
        {"photoIds": ["d", "e"]},
    ]}}
    parsed = PredefinedLayoutInput.from_request(content)

    assert isinstance(parsed, PredefinedLayoutInput)
    assert len(parsed.spreads) == 2
    assert parsed.spreads[0].photo_ids == ["a", "b", "c"]
    assert parsed.spreads[1].photo_ids == ["d", "e"]
    # photos-only: forward-compat fields stay None
    assert parsed.spreads[0].layout_id is None
    assert parsed.spreads[0].left_photo_ids is None
    assert parsed.spreads[0].right_photo_ids is None
    assert parsed.first_page_photo_ids is None
    assert parsed.last_page_photo_ids is None


def test_parses_forward_compat_fields():
    content = {"predefinedLayout": {"spreads": [
        {"photoIds": ["a", "b"], "layoutId": 42,
         "leftPhotoIds": ["a"], "rightPhotoIds": ["b"]},
    ]}}
    spread = PredefinedLayoutInput.from_request(content).spreads[0]

    assert spread.layout_id == 42
    assert spread.left_photo_ids == ["a"]
    assert spread.right_photo_ids == ["b"]


def test_parses_cover_fields():
    content = {"predefinedLayout": {
        "spreads": [{"photoIds": ["a", "b"]}],
        "firstPagePhotoIds": ["cover1"],
        "lastPagePhotoIds": ["cover2"],
    }}
    parsed = PredefinedLayoutInput.from_request(content)

    assert parsed.first_page_photo_ids == ["cover1"]
    assert parsed.last_page_photo_ids == ["cover2"]


def test_all_photo_ids_dedups_preserving_order():
    parsed = PredefinedLayoutInput(
        spreads=[PredefinedSpread(photo_ids=["a", "b"]),
                 PredefinedSpread(photo_ids=["b", "c"])],  # 'b' repeats across spreads
        first_page_photo_ids=["cover1", "a"],              # 'a' repeats vs a spread
        last_page_photo_ids=["cover2"],
    )
    assert parsed.all_photo_ids() == ["a", "b", "c", "cover1", "cover2"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")

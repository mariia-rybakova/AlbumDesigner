"""The keywords that are instructions to this service, not descriptions.

A vectorised photo carries the keywords its catalogue was filed under, read
onto the photo table as ``Col.KEYWORDS`` by ``ingest.photo_metadata``. Most of
them describe the picture -- a place, a person, an event -- and the album has
nothing to say about those. Two are addressed to the album:

``notforalbum``
    Keep this photograph out. Matched loosely, because it is typed by hand in
    whatever shape the photographer's catalogue encourages: case is folded and
    spaces, hyphens and underscores are dropped, so ``Not For Album``,
    ``not-for-album`` and ``NOT_FOR_ALBUM`` are one keyword. It is the highest
    authority in the pipeline -- a photo carrying it is removed before anything
    counts, ranks or groups the gallery, so no later rule can reinstate it, not
    even ``album`` on the same photo.

``album``
    Put this photograph in. Matched **exactly** -- the whole keyword, case
    folded and nothing else -- so ``album`` and ``Album`` qualify while
    ``wedding album``, ``albums`` and ``notforalbum`` do not. The loose match
    is reserved for the exclusion on purpose: a false positive there costs one
    photograph the album would have liked, while a loose ``album`` would sweep
    in every keyword a photographer ever typed the word inside.

Both predicates take whatever the column holds -- a list, a tuple, or the
``[]`` the reader writes for a photo whose payload named no keywords -- and
answer False for anything else rather than raising. A gallery that was never
vectorised has no keyword column at all and neither rule fires.
"""

import re

#: The exclusion, after folding. See the module docstring for why it is loose.
NOT_FOR_ALBUM = "notforalbum"

#: The priority marker, matched whole and case-insensitively, nothing folded.
FOR_ALBUM = "album"

#: What the loose match ignores: case, and every run of whitespace, hyphen or
#: underscore anywhere in the keyword.
_SEPARATORS = re.compile(r"[\s\-_]+")


def fold(keyword) -> str:
    """``'Not For Album'`` -> ``'notforalbum'``. Non-strings fold to ``''``."""
    if not isinstance(keyword, str):
        return ""
    return _SEPARATORS.sub("", keyword).casefold()


def _keywords(value):
    """The keyword list, or an empty one for anything that is not a list."""
    return value if isinstance(value, (list, tuple, set)) else ()


def is_not_for_album(keywords) -> bool:
    """Whether the photographer marked this photograph as one to leave out."""
    return any(fold(word) == NOT_FOR_ALBUM for word in _keywords(keywords))


def is_for_album(keywords) -> bool:
    """Whether the photographer marked this photograph as one to put in.

    Says nothing about ``notforalbum`` on the same photo. Nothing has to ask,
    because such a photo is gone from the table before this is ever called --
    see ``enrich.album_keywords``.
    """
    return any(isinstance(word, str) and word.strip().casefold() == FOR_ALBUM
               for word in _keywords(keywords))


# --------------------------------------------------------------------------
# What `album` is worth
# --------------------------------------------------------------------------

#: The column the keyword list arrives in. Spelled out rather than imported
#: from `pipeline.contracts`, which imports from here.
KEYWORDS_COLUMN = "keywords"

#: How far a photograph keyworded `album` is lifted above its peers, added to
#: `total_score` after the weighted blend.
#:
#: The blend normalises every component to 0..1 and its weights to sum to 1, so
#: a score lives in 0..1 and this is a quarter of the whole range: a marked
#: photograph beats any unmarked one within 0.25 of it, and loses to one that
#: is better than that by more. That is the shape asked for -- a priority, not
#: a guarantee. The per-category budget is untouched, so a marked photograph in
#: an over-subscribed category can still miss, and an album does not grow
#: because a gallery was heavily keyworded.
#:
#: Added rather than blended in as a seventh weight, which would have meant
#: renormalising the other six and moving every score in every gallery. This
#: way **a photograph with no `album` keyword scores exactly what it scored
#: before this rule existed**, and a gallery with none is bit-identical -- which
#: is what makes the rule bisectable against a rendered album.
#:
#: Raise it towards 1.0 to make the keyword close to absolute, or set it to 0
#: to turn the rule off without removing it.
FOR_ALBUM_BONUS = 0.25


def for_album_lift(frame, bonus: float = None):
    """Per-row score lift for a photo frame: ``bonus`` where ``album``, else 0.

    Zero everywhere for a frame with no keyword column -- a gallery that was
    never vectorised -- so no caller needs to ask whether this gallery has
    keywords at all.
    """
    import pandas as pd

    # Read at call time, not bound as a default, so setting the constant to 0
    # really does turn the rule off.
    bonus = FOR_ALBUM_BONUS if bonus is None else bonus
    if KEYWORDS_COLUMN not in getattr(frame, 'columns', ()) or not bonus:
        return pd.Series(0.0, index=frame.index)
    return frame[KEYWORDS_COLUMN].map(is_for_album).astype(float) * float(bonus)


"""Reading the gallery's own scene list, for what it says about a photograph.

`ptinfra.utils.gallery.Gallery` decodes `gallery.json.txt` into scenes, and one
scene attribute is addressed to the album: ``Scene.isHighlights``, which is
``userJobId > 0``. A Highlights scene is the reel the photographer built to
show the client, so every photograph in it is one they singled out.

It is a *scene* flag, and what the album wants is a *photograph* flag, which is
what this module turns it into. A Highlights scene holds no photographs of its
own -- it lists `sharedPhotoIds`, references to photographs that also live in
an ordinary scene -- so being in Highlights is a second thing true of a
photograph, never a separate photograph.

Nothing here infers. `src/core/anachronism_scene_detection.py` separately
guesses at highlight-*like* scenes from their names, for galleries where the
reel was built by hand rather than as a user job; that is a different question
and deliberately not mixed in here.

`src/core/time_processing.py` reads the same flag for the opposite purpose --
it *skips* Highlights scenes when building the artificial timeline, because a
reel gathers shots from across the whole day and would wreck the ordering.
Both readings are right; they are asking different things.
"""

from ptinfra.utils.gallery import Gallery


def highlight_photo_ids(project_base_url, logger=None):
    """The ids of every photograph in a Highlights scene.

    Returns an empty set when the gallery has no Highlights reel, and also
    when the scene list cannot be read at all -- that is a gallery whose
    photographs are simply unmarked, which is the same thing the album does
    with a gallery that was never vectorised. The caller is a best-effort
    substage, so a blob that is unreachable costs the marks, not the album.
    """
    try:
        gallery = Gallery(project_base_url)
    except Exception as exc:  # noqa: BLE001 - boundary: a reel is not an album
        if logger:
            logger.warning(
                f"Could not read the scene list for Highlights: {exc}")
        return set()

    ids = set()
    reels = 0
    for scene in gallery.scenes:
        if not getattr(scene, 'isHighlights', False):
            continue
        reels += 1
        ids.update(photo.photoId for photo in scene.photos)

    if logger:
        logger.info(f"Highlights: {len(ids)} photos across {reels} scene(s)")
    return ids

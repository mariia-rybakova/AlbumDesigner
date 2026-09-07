"""Build an album request for a gallery that never had one.

Album Designer is only reachable from a **wedding** gallery, so no non-wedding
album request exists to replay -- production logs show zero non-wedding album
runs over 30 days, and that is not a gap in the data, it is the product. But
`select.narrator` exists precisely for non-wedding galleries, so testing it
needs a request that was never made: a real gallery, a real product design, and
a synthesised envelope around them.

What is real and what is invented
---------------------------------
Real: the ``projectId``, the ``base_url`` (resolved from ``pictime.projectdal``
exactly as the narrator's downloader does), the photo ids (read from the
gallery's own manifest, so they are the ids AD will find), and the product
design.

Invented: the queue envelope -- ``replyQueueName``, ``userJobId``,
``conditionId``, ``fulfillerId`` -- and the ``aiMetadata`` block. None of it is
sent anywhere; it exists so `read_messages` and the pipeline see the shape they
expect.

``designInfo`` is **inlined** rather than pointed at by
``designInfoTempLocation``. Those temp blobs expire within days, which is the
trap that blocked this integration before: every saved request in
``files/test_requests/`` points at one that has gone, so ``designInfo`` reads
back as null and the run dies in layout. `read_layouts_data` only fetches the
blob when ``designInfo`` is None, so inlining it makes the fixture permanent.

``aiMetadata.photoIds`` is ``[]``, not null, on purpose. Null means *manual* --
the user assembled the album -- and `select.route` then skips selection
entirely. An empty list means "an AI request in which the user picked nothing",
which is the case the narrator is for.

Usage
-----
    # a Family gallery from the narrator's corpus, with its product design
    python tools/dummy_request.py --project-id 19778020 \
        --design-info C:/path/to/designinfo_product919.json \
        --name 19778020_family_dummy

The projectdal credential is read from ``$PICTIME_MONGO`` (or ``--pictime-mongo``)
and is never defaulted in this file: it carries live reader credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Gallery category enum values, for labelling the request. AD routes on the
#: content classifier rather than this, so it is documentation, not control.
CATEGORIES = {'family': 4, 'portraits': 26, 'seniors': 38, 'graduation': 42}


def build_base_url(project_id: int, path_token: str, public_storage_id: int) -> str:
    """The deterministic blob layout, same scheme the narrator's downloader uses."""
    return (f"ptstorage_{public_storage_id}://pictures/"
            f"{project_id // 1000000}/{(project_id // 1000) % 1000}/"
            f"{project_id}/{path_token}")


def resolve_base_url(project_id: int, connection: str) -> Optional[str]:
    """projectId -> base_url via one `pictime.projectdal` lookup.

    `projectcategorystatsdal` knows the category but not the path token, so this
    second collection is what makes a project id sufficient on its own.
    """
    from pymongo import MongoClient

    doc = MongoClient(connection)['pictime']['projectdal'].find_one(
        {'_id': int(project_id)}, {'pathToken': 1, 'publicStorageId': 1})
    if not doc or doc.get('pathToken') is None or doc.get('publicStorageId') is None:
        return None
    return build_base_url(project_id, doc['pathToken'], int(doc['publicStorageId']))


def gallery_photo_ids(base_url: str) -> List[int]:
    """Every photo id in the gallery, from its own manifest.

    Read rather than taken from the metadata protobufs so the list is exactly
    what AD will enumerate -- a proto can carry ids for photos since deleted.
    """
    from ptinfra.utils.gallery import Gallery

    gallery = Gallery(base_url)
    return [int(p.photoId) for scene in gallery.scenes for p in scene.photos]


def build_request(project_id: int, base_url: str, photo_ids: List[int],
                  design_info: dict, category: int = 4, density: int = 3,
                  store_id: int = 32) -> dict:
    """The synthesised request. See the module docstring for real vs invented."""
    return {
        'replyQueueName': 'aigeneratealbumresponsedto',
        'storeId': store_id,
        'accountId': 0,
        'projectId': int(project_id),
        'fulfillerId': 0,
        'userId': 0,
        'userJobId': 0,
        'base_url': base_url,
        'photos': list(photo_ids),
        'projectCategory': int(category),
        'compositionPackageId': -1,
        # Inlined, so this fixture does not rot when the temp blob expires.
        'designInfo': design_info,
        'designInfoTempLocation': None,
        'aiMetadata': {
            # Empty, not null: an AI request where the user chose nothing.
            'photoIds': [],
            'focus': [],
            'personIds': [],
            'subjects': [],
            'density': int(density),
        },
        # No per-photo ratings: the user never rated this gallery. AD treats an
        # empty list as "no ratings", which is the truth here.
        'rating': [],
        'ratingTempLocation': None,
        'conditionId': f'DUMMY_{project_id}_narrator_test',
        'timedOut': False,
        'dependencyDeleted': False,
        'retryCount': 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', type=int,
                        help="Resolve base_url and photos from pictime.projectdal. "
                             "Needs that Mongo reachable; use --from-request otherwise.")
    parser.add_argument('--from-request',
                        help="Take projectId, base_url and photos from a saved request "
                             "instead of resolving them. The way in when "
                             "pictime.projectdal is unreachable -- it lives on a "
                             "different host from the aimongo AD uses, and is not "
                             "routable from every network.")
    parser.add_argument('--design-info', required=True,
                        help="A designInfo JSON to inline (a saved product design).")
    parser.add_argument('--name', help="Fixture name. Default: <projectId>_dummy")
    parser.add_argument('--category', default='family',
                        help="family|portraits|seniors|graduation, or an integer.")
    parser.add_argument('--density', type=int, default=3)
    parser.add_argument('--store-id', type=int, default=32)
    parser.add_argument('--pictime-mongo',
                        help="pictime connection string. Default: $PICTIME_MONGO.")
    parser.add_argument('--settings',
                        default=os.environ.get(
                            'HostingSettingsPath',
                            '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt'))
    args = parser.parse_args()

    if not args.project_id and not args.from_request:
        print("need --project-id or --from-request")
        return 2

    from tools import local_request

    if args.from_request:
        # The gallery's real coordinates, lifted off a request that really
        # happened. Only the envelope and aiMetadata are rebuilt below.
        source = local_request.load_request(args.from_request)
        project_id = int(source['projectId'])
        base_url = source['base_url']
        photo_ids = [int(p) for p in (source.get('photos') or [])]
        store_id = int(source.get('storeId', args.store_id))
        print(f"source      {args.from_request} (real request)")
    else:
        connection = args.pictime_mongo or os.environ.get('PICTIME_MONGO')
        if not connection:
            print("need the pictime connection string in $PICTIME_MONGO or --pictime-mongo")
            return 2
        import ptinfra
        ptinfra.intialize('DummyRequest', args.settings)
        project_id = args.project_id
        store_id = args.store_id
        base_url = resolve_base_url(project_id, connection)
        if not base_url:
            print(f"could not resolve base_url for {project_id} in pictime.projectdal")
            return 1
        photo_ids = gallery_photo_ids(base_url)

    print(f"projectId   {project_id}")
    print(f"base_url    {base_url}")
    print(f"photos      {len(photo_ids)}")
    if not photo_ids:
        print("no photos to compose from")
        return 1

    with open(args.design_info, 'r', encoding='utf-8') as handle:
        design_info = json.load(handle)
    parts = list((design_info.get('parts') or {}))
    print(f"designInfo  product {design_info.get('productId')}, "
          f"{len(design_info.get('designs') or {})} designs, parts {parts}, "
          f"pages {design_info.get('minPages')}-{design_info.get('maxPages')}")

    category = CATEGORIES.get(str(args.category).lower())
    if category is None:
        category = int(args.category)

    request = build_request(project_id, base_url, photo_ids, design_info,
                            category=category, density=args.density,
                            store_id=store_id)
    name = args.name or f"{project_id}_dummy"
    path = local_request.save_request(request, name)
    print(f"wrote       {path}")
    print(f"replay with: process_gallery.py dataset output --request {name}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

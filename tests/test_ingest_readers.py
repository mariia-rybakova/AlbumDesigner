"""Golden tests for the gallery-asset readers.

The readers now decode through ptinfra (`ptinfra.proto.pb` schemas,
`ptinfra.read_stage.load_versioned`, `ptinfra.exporter.pai_writer.parse_pai`)
instead of hand-rolled parsing against a vendored copy of the protos. These
tests build synthetic blobs with known content, serve them through a patched
PTFile, and assert the DataFrames the readers return are exactly what the rest
of the pipeline expects.

    python -m pytest tests/test_ingest_readers.py -v
    python tests/test_ingest_readers.py
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptinfra.exporter.pai_writer import serialize_pai_v3  # noqa: E402
from ptinfra.proto.pb import BGSegmentation_pb2, ContentCluster_pb2  # noqa: E402
from ptinfra.proto.pb import FaceVector_pb2, PersonInfo_pb2  # noqa: E402
from ptinfra.proto.pb import PersonVector_pb2, SocialCircle_pb2  # noqa: E402

import utils.read_protos_files as readers  # noqa: E402

PHOTO_A, PHOTO_B = 5001, 5002
IDENTITY_A, IDENTITY_B = 11, 22


class _NullLogger:
    def __getattr__(self, _name):
        return lambda *a, **k: None


LOGGER = _NullLogger()


# --------------------------------------------------------------------------
# Synthetic blobs
# --------------------------------------------------------------------------


def face_blob() -> bytes:
    msg = FaceVector_pb2.FaceVectorMessageWrapper()
    msg.v1.accountId, msg.v1.storeId, msg.v1.ModelVersion = 7, 32, 2
    for photo_id, n_faces in ((PHOTO_A, 2), (PHOTO_B, 1)):
        photo = msg.v1.photos.add()
        photo.photoId = photo_id
        for i in range(n_faces):
            face = photo.faces.add()
            face.id = f"f{photo_id}-{i}"
            face.bbox.x1, face.bbox.y1 = 0.1 * (i + 1), 0.2 * (i + 1)
            face.bbox.x2, face.bbox.y2 = 0.5 * (i + 1), 0.6 * (i + 1)
            face.embedding = np.arange(4, dtype=np.float32).tobytes()
            face.probability = 0.9
    return msg.SerializeToString()


def bg_blob() -> bytes:
    msg = BGSegmentation_pb2.PhotoBGSegmentationMessageWrapper()
    msg.v1.accountId, msg.v1.storeId, msg.v1.modelVersion = 7, 32, 3
    for i, photo_id in enumerate((PHOTO_A, PHOTO_B)):
        photo = msg.v1.photos.add()
        photo.photoId = photo_id
        photo.blobCentroid.x, photo.blobCentroid.y = 0.4 + i * 0.1, 0.5
        photo.blobDiameter = 0.3 + i * 0.1
        photo.dateTaken = 1_700_000_000 + i * 60
        photo.sceneOrder = i
        photo.orderInScene = i * 2
        photo.aspectRatio = 1.5 if i == 0 else 0.66
        photo.colorEnum = 1 if i == 0 else 0
    return msg.SerializeToString()


def cluster_blob() -> bytes:
    msg = ContentCluster_pb2.ContentClusterMessageWrapper()
    msg.v1.accountId, msg.v1.storeId = 7, 32
    for i, photo_id in enumerate((PHOTO_A, PHOTO_B)):
        photo = msg.v1.photos.add()
        photo.photoId = photo_id
        photo.imageClass = 2 + i
        photo.clusterId = 4 + i
        photo.clusterClass = 1 + i
        photo.selectionOrder = 10 + i
        photo.selectionScore = 0.75 - i * 0.25
    return msg.SerializeToString()


def persons_blob() -> bytes:
    msg = PersonInfo_pb2.PersonInfoMessageWrapper()
    msg.v1.accountId, msg.v1.storeId, msg.v1.clusteringVersion = 7, 32, 1
    # IDENTITY_A is in both photos, IDENTITY_B in one -> main_persons order.
    for identity_id, age, gender, photos in (
        (IDENTITY_A, 30.0, 1, (PHOTO_A, PHOTO_B)),
        (IDENTITY_B, 34.0, 0, (PHOTO_A,)),
    ):
        identity = msg.v1.identities.add()
        identity.identityNumeralId = identity_id
        identity.personInfo.age = age
        identity.personInfo.gender = gender
        identity.personInfo.bestPhoto = photos[0]
        for photo_id in photos:
            image = identity.personInfo.imagesInfo.add()
            image.photoId = photo_id
            image.faceBbox.x1, image.faceBbox.y1 = 0.1, 0.1
            image.faceBbox.x2, image.faceBbox.y2 = 0.4, 0.4
    return msg.SerializeToString()


def person_vector_blob() -> bytes:
    msg = PersonVector_pb2.PersonVectorMessageWrapper()
    msg.v1.accountId, msg.v1.storeId = 7, 32
    msg.v1.bodyModelVersion, msg.v1.bodyDetectVersion = 2, 1
    for photo_id, n_bodies in ((PHOTO_A, 2), (PHOTO_B, 1)):
        photo = msg.v1.photos.add()
        photo.photoId = photo_id
        for i in range(n_bodies):
            body = photo.bodies.add()
            body.id = f"b{photo_id}-{i}"
            body.bbox.x1, body.bbox.y1 = 0.05, 0.05
            body.bbox.x2, body.bbox.y2 = 0.7, 0.9
            body.conf = 0.8
            body.embedding = np.arange(4, dtype=np.float32).tobytes()
    return msg.SerializeToString()


def social_blob() -> bytes:
    msg = SocialCircle_pb2.SocialCircleMessageWrapper()
    msg.v1.accountId, msg.v1.storeId = 7, 32
    circle = msg.v1.socialCircles.add()
    circle.identityNumeralId.extend([IDENTITY_A, IDENTITY_B])
    return msg.SerializeToString()


def pai_blob() -> bytes:
    return serialize_pai_v3(2, [
        (PHOTO_A, np.arange(8, dtype=np.float32)),
        (PHOTO_B, np.arange(8, 16, dtype=np.float32)),
    ])


BLOBS = {
    'ai_face_vectors.pb': face_blob,
    'bg_segmentation.pb': bg_blob,
    'content_cluster.pb': cluster_blob,
    'persons_info.pb': persons_blob,
    'ai_person_vectors.pb': person_vector_blob,
    'social_circles.pb': social_blob,
    'ai_search_matrix.pai': pai_blob,
}


# --------------------------------------------------------------------------
# PTFile stand-in
# --------------------------------------------------------------------------


class FakePTFile:
    """Serves the synthetic blobs by filename; anything else is 'absent'."""

    #: filenames to report as missing, for the missing_ok paths
    missing: set = set()

    def __init__(self, url):
        self.url = url
        self.name = str(url).replace('\\', '/').rsplit('/', 1)[-1]

    def exists(self):
        return self.name in BLOBS and self.name not in self.missing

    def read_blob(self):
        if not self.exists():
            raise FileNotFoundError(self.url)
        return BLOBS[self.name]()


@contextmanager
def fake_blobs(missing=()):
    FakePTFile.missing = set(missing)
    try:
        # load_versioned imports PTFile from azurev12 at call time; the readers
        # that bypass it (the PAI one) use the module-level import.
        with mock.patch('ptinfra.azurev12.pt_file.PTFile', FakePTFile), \
             mock.patch.object(readers, 'PTFile', FakePTFile):
            yield
    finally:
        FakePTFile.missing = set()


BASE = 'ptstorage_32://pictures/7/1/1/abc'


def path(name):
    return f'{BASE}/{name}'


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_face_reader():
    with fake_blobs():
        df = readers.get_faces_info(path('ai_face_vectors.pb'), LOGGER)

    assert list(df.columns) == ['image_id', 'n_faces', 'faces_info']
    assert list(df['image_id']) == [PHOTO_A, PHOTO_B]
    assert list(df['n_faces']) == [2, 1]
    # Cropping reads bbox attributes off these objects; keep them proto-shaped.
    first_face = df.loc[0, 'faces_info'][0]
    assert (first_face.bbox.x1, first_face.bbox.y2) == (
        np.float32(0.1).item(), np.float32(0.6).item()
    )


def test_photo_meta_reader():
    with fake_blobs():
        df = readers.get_photo_meta(path('bg_segmentation.pb'), LOGGER)

    assert list(df['image_id']) == [PHOTO_A, PHOTO_B]
    assert list(df['image_time']) == [1_700_000_000, 1_700_000_060]
    assert list(df['scene_order']) == [0, 1]
    assert list(df['image_color']) == [1, 0]
    assert list(df['image_orientation']) == ['landscape', 'portrait']
    assert df.loc[0, 'background_centroid'].x == np.float32(0.4).item()
    assert round(df.loc[1, 'diameter'], 5) == round(np.float32(0.4).item(), 5)


def test_cluster_reader():
    with fake_blobs():
        df = readers.get_clusters_info(path('content_cluster.pb'), LOGGER)

    assert list(df.columns) == [
        'image_id', 'image_class', 'cluster_label', 'cluster_class', 'ranking', 'image_order'
    ]
    assert list(df['image_class']) == [2, 3]
    assert list(df['cluster_label']) == [4, 5]
    assert list(df['image_order']) == [10, 11]
    assert round(df.loc[0, 'ranking'], 5) == 0.75


def test_persons_reader():
    with fake_blobs():
        persons_df, details_df = readers.get_persons_ids(path('persons_info.pb'), LOGGER)

    by_photo = dict(zip(persons_df['image_id'], persons_df['persons_ids']))
    assert sorted(by_photo[PHOTO_A]) == [IDENTITY_A, IDENTITY_B]
    assert by_photo[PHOTO_B] == [IDENTITY_A]
    # main_persons: the two most-photographed identities, same on every row.
    assert all(set(v) == {IDENTITY_A, IDENTITY_B} for v in persons_df['main_persons'])
    assert set(details_df['identity_id']) == {IDENTITY_A, IDENTITY_B}
    assert details_df.set_index('identity_id').loc[IDENTITY_A, 'gender'] == 1


def test_person_vector_reader():
    with fake_blobs():
        df = readers.get_person_vectors(path('ai_person_vectors.pb'), LOGGER)

    assert list(df['image_id']) == [PHOTO_A, PHOTO_B]
    assert list(df['number_bodies']) == [2, 1]
    assert df.loc[0, 'bodies_info'][0].bbox.x2 == np.float32(0.7).item()


def test_social_circle_reader():
    with fake_blobs():
        df = readers.get_social_circle(path('social_circles.pb'), LOGGER)

    assert list(df['identity_ids']) == [[IDENTITY_A, IDENTITY_B]]
    assert list(df['num_ids']) == [2]


def test_embeddings_reader():
    with fake_blobs():
        df = readers.get_image_embeddings(path('ai_search_matrix.pai'), LOGGER)

    assert list(df['image_id']) == [PHOTO_A, PHOTO_B]
    assert list(df['model_version']) == [2, 2]
    np.testing.assert_array_equal(df.loc[0, 'embedding'], np.arange(8, dtype=np.float32))
    assert df.loc[0, 'embedding'].dtype == np.float32


def test_missing_blob_returns_none():
    """The readers that guarded on exists() still return None, not an error."""
    for name, reader in (
        ('persons_info.pb', readers.get_persons_ids),
        ('content_cluster.pb', readers.get_clusters_info),
        ('ai_person_vectors.pb', readers.get_person_vectors),
        ('social_circles.pb', readers.get_social_circle),
    ):
        with fake_blobs(missing={name}):
            assert reader(path(name), LOGGER) is None, f"{name} should read as absent"


def test_load_gallery_assets_end_to_end():
    """The whole ingest substage, against synthetic blobs."""
    with fake_blobs():
        photos, person_details, social_circles, error = readers.load_gallery_assets(
            BASE, LOGGER, clip_df=None
        )

    assert error is None
    assert list(photos['image_id']) == [PHOTO_A, PHOTO_B]

    # Every column the enrich substages and the selection stage rely on.
    required = {
        'image_id', 'embedding', 'model_version', 'image_class', 'cluster_label',
        'cluster_class', 'ranking', 'image_order', 'persons_ids', 'main_persons',
        'n_faces', 'faces_info', 'number_bodies', 'bodies_info', 'image_time',
        'image_as', 'image_color', 'image_orientation', 'scene_order',
        'background_centroid', 'diameter',
    }
    missing = required - set(photos.columns)
    assert not missing, f"load_gallery_assets dropped columns: {sorted(missing)}"

    assert photos['image_class'].dtype == 'Int64'
    assert social_circles is not None and len(social_circles) == 1
    assert person_details is not None and len(person_details) == 2


def test_pipeline_ingest_substage_uses_the_readers():
    """The registered substage produces what it declares."""
    from src.pipeline import AlbumContext, Col
    from src.pipeline.registry import get

    context = AlbumContext(logger=LOGGER, project_url=BASE)
    with fake_blobs():
        context = get('ingest.gallery_assets')()(context)

    assert not context.failed, context.error
    assert len(context.photos) == 2
    assert context.facts.model_version == 2
    assert context.social_circles is not None
    # the substage's declared provides must actually hold
    assert context.missing(get('ingest.gallery_assets').provides) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} reader tests passed")

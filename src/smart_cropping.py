from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from utils.configs import CONFIGS


def cropWeight(tl, imageShape, foregroundMask=None, faceMask=None, aspectRatio=1,
               center_weight=200, face_weight=150, size_weight=1000):
    tl = np.array(tl).astype(np.int32)  # tl = [x, y]
    width, height = imageShape  # imageShape = [width, height]
    max_w = width - tl[0]
    max_h = height - tl[1]

    # Calculate the crop dimensions based on the aspect ratio
    if aspectRatio == 1:
        # For aspect ratio 1, use the largest possible square
        w = h = min(max_w, max_h)
    else:
        if max_w / aspectRatio <= max_h:
            w = int(max_w)
            h = int(max_w / aspectRatio)
        else:
            h = int(max_h)
            w = int(max_h * aspectRatio)

    # Ensure w and h are positive integers
    if w <= 0 or h <= 0:
        return float('inf'), 0, 0

    # Adjust w and h if they exceed image boundaries
    if tl[0] + w > width:
        w = width - tl[0]
    if tl[1] + h > height:
        h = height - tl[1]

    # Compute size ratio
    area = w * h
    image_area = width * height
    size_ratio = area / image_area

    # Penalties
    if aspectRatio == 1:
        size_penalty = 0
        center_penalty = 0
    else:
        size_penalty = size_weight * (1 - size_ratio)
        # Calculate the center of the crop and the image
        center = np.array((tl[0] + w / 2, tl[1] + h / 2))  # center = [x, y]
        o_center = np.array([width / 2, height / 2])  # image center = [x, y]
        center_penalty = np.linalg.norm(o_center - center) * center_weight

    fl_tl = np.floor(tl).astype(int)
    # Access arrays as [rows, cols] = [y, x]
    if fl_tl[1] + h > foregroundMask.shape[0] or fl_tl[0] + w > foregroundMask.shape[1]:
        return float('inf'), 0, 0

    croppedForeground = foregroundMask[fl_tl[1]:fl_tl[1] + h, fl_tl[0]:fl_tl[0] + w]
    fl_weight = np.abs(np.sum(foregroundMask) - np.sum(croppedForeground))

    if faceMask is not None:
        croppedFace = faceMask[fl_tl[1]:fl_tl[1] + h, fl_tl[0]:fl_tl[0] + w]
        fl_weight += np.abs(np.sum(faceMask) - np.sum(croppedFace)) * face_weight

    # Total weight
    total_weight = fl_weight + size_penalty + center_penalty

    return total_weight, w, h

def cropStep(xRange, yRange, foregroundMask, faceMask=None, aspectRatio=1, steps=10):
    x = np.linspace(xRange[0], xRange[1], steps + 1)[:steps].astype(int)
    y = np.linspace(yRange[0], yRange[1], steps + 1)[:steps].astype(int)

    cr_weights = np.zeros((len(x), len(y)))
    w = np.zeros((len(x), len(y)))
    h = np.zeros((len(x), len(y)))

    for xi, x_val in enumerate(x):
        for yi, y_val in enumerate(y):
            temp_weight, temp_w, temp_h = cropWeight(
                [x_val, y_val],
                [foregroundMask.shape[1], foregroundMask.shape[0]],
                foregroundMask,
                faceMask,
                aspectRatio
            )
            cr_weights[xi, yi] = temp_weight
            w[xi, yi] = temp_w
            h[xi, yi] = temp_h

    m = np.unravel_index(np.argmin(cr_weights), cr_weights.shape)
    return m, x, y, w[m], h[m]



def crop_find(foregroundMask, faceMask=None, aspectRatio=1, steps=10):
    s_min = [0, 0]  # [x_min, y_min]
    s_max = [foregroundMask.shape[1], foregroundMask.shape[0]]  # [width, height]

    while s_max[0] - s_min[0] > 10 and s_max[1] - s_min[1] > 10:
        m, x, y, w, h = cropStep(
            [s_min[0], s_max[0]],
            [s_min[1], s_max[1]],
            foregroundMask,
            faceMask,
            aspectRatio,
            steps
        )
        s_min[0] = x[m[0]]
        s_min[1] = y[m[1]]
        if m[0] < len(x) - 1:
            s_max[0] = x[m[0] + 1]
        else:
            s_max[0] = x[m[0]]
        if m[1] < len(y) - 1:
            s_max[1] = y[m[1] + 1]
        else:
            s_max[1] = y[m[1]]

    return s_min, [s_min[0] + w, s_min[1] + h], int(w), int(h)

def process_cropping(ar, faces, centroid, diameter, box_aspect_ratio, min_dim=1000, face_extension=2):
    #print("ar",ar, faces, centroid, diameter, box_aspect_ratio, min_dim, face_extension)

    min_face_x = 1
    min_face_y = 1
    max_face_x = 0
    max_face_y = 0

    for face in faces:
        bbox = face.bbox

        min_face_x = min(min_face_x, bbox.x1)
        min_face_y = min(min_face_y, bbox.y1)
        max_face_x = max(max_face_x, bbox.x2)
        max_face_y = max(max_face_y, bbox.y2)

    face_width = max_face_x - min_face_x
    face_height = max_face_y - min_face_y

    if len(faces) == 0 and box_aspect_ratio == 1:
        if ar > 1:
            h=1
            w = 1/ar
            y=0
            x = centroid.x - w/2
            x= max(0,x)
            x=min(x,1-w)


        else:
            h = ar
            w = 1
            x = 0
            y = centroid.y - h / 2
            y = max(0, y)
            y = min(y, 1 - h)

        return x, y, w, h

    elif box_aspect_ratio == 1 and ar>1 and face_width< 1/ar:
        h = 1
        w = 1 / ar
        y = 0
        x = (min_face_x + max_face_x)/2 - w / 2
        x = max(0, x)
        x = min(x, 1 - w)

        return x, y, w, h

    elif box_aspect_ratio == 1 and ar<1 and face_height< ar:
        h = ar
        w = 1
        x = 0
        y = (min_face_y + max_face_y)/2 - h / 2
        y = max(0, y)
        y = min(y, 1 - h)

        return x, y, w, h

    else:

        if ar > 1:
            mask = np.zeros((min_dim, int(ar * min_dim)), dtype=np.uint8)
        else:
            mask = np.zeros((int(min_dim / ar), min_dim), dtype=np.uint8)

        # Correct the centroid mapping
        mask = cv2.circle(
            mask,
            (int(centroid.x * mask.shape[1]), int(centroid.y * mask.shape[0])),  # (x, y)
            int(diameter / 2 * mask.shape[0]),
            255,
            -1
        )

        face_mask = None
        if len(faces) !=0:
            face_mask = np.zeros_like(mask, dtype=np.uint8)
            if not isinstance(faces, list):
                faces = list(faces)

            for face in faces:
                bbox = face.bbox
                # x1 = int(bbox.x1 * face_mask.shape[1])
                # y1 = int(bbox.y1 * face_mask.shape[0])
                # x2 = int(bbox.x2 * face_mask.shape[1])
                # y2 = int(bbox.y2 * face_mask.shape[0])

                x1 = int(bbox.x1*mask.shape[1])
                y1 = int(bbox.y1*mask.shape[0])
                x2 = int(bbox.x2*mask.shape[1])
                y2 = int(bbox.y2*mask.shape[0])
                bbox_w = (x2 - x1) * face_extension
                bbox_h = (y2 - y1) * face_extension

                # x1 = int(max(0, x1 - bbox_w / 2))
                # y1 = int(max(0, y1 - bbox_h / 2))
                # x2 = int(min(mask.shape[1], x2 + bbox_w / 2))
                # y2 = int(min(mask.shape[0], y2 + bbox_h / 2))

                x1 = int(max(0, min(face_mask.shape[1] - 1, x1 - bbox_w / 2)))
                y1 = int(max(0, min(face_mask.shape[0] - 1, y1 - bbox_h / 2)))
                x2 = int(max(0, min(face_mask.shape[1] - 1, x2 + bbox_w / 2)))
                y2 = int(max(0, min(face_mask.shape[0] - 1, y2 + bbox_h / 2)))

                # Access arrays as [rows, cols] = [y, x]
                face_mask[y1:y2, x1:x2] = 255

        s_min, s_max, w, h = crop_find(
            mask,
            faceMask=face_mask,
            aspectRatio=box_aspect_ratio,
            steps=10
        )

        # Ensure the crop dimensions are within the image boundaries
        s_min[0] = max(0, s_min[0])
        s_min[1] = max(0, s_min[1])
        w = min(w, mask.shape[1] - s_min[0])
        h = min(h, mask.shape[0] - s_min[1])

        # Return normalized coordinates
        return s_min[0] / mask.shape[1], s_min[1] / mask.shape[0], w / mask.shape[1], h / mask.shape[0]


def process_crop_images(q,df):
    results = []
    for _, row in df.iterrows():
        cropped_x, cropped_y, cropped_w, cropped_h = process_cropping(
            float(row['image_as']),
            row['faces_info'],
            row['background_centroid'],
            float(row['diameter']),
            1
        )
        # Store the results in a dictionary to update the DataFrame later
        results.append({
            'image_id': row['image_id'],
            'cropped_x': cropped_x,
            'cropped_y': cropped_y,
            'cropped_w': cropped_w,
            'cropped_h': cropped_h
        })
    cropped_df = pd.DataFrame(results)
    q.put(cropped_df)


if __name__ == "__main__":
    project_id = 46229129
    from multiprocessing import Queue
    q = Queue()
    input_path = rf'C:\Users\ZivRotman\PycharmProjects\logAnalysis\galleries_pbs2\{project_id}'
    from ptinfra.proto.pb import FaceVector_pb2 as face_vector
    from ptinfra.proto.pb import BGSegmentation_pb2 as meta_vector
    import os
    
    face_file = os.path.join(input_path, "ai_face_vectors.pb")

    faces_info_bytes = open(face_file, 'rb').read()
    face_descriptor = face_vector.FaceVectorMessageWrapper()
    face_descriptor.ParseFromString(faces_info_bytes)

    if face_descriptor.WhichOneof("versions") == 'v1':
        message_data = face_descriptor.v1

    images_photos = message_data.photos

    photo_ids = []
    num_faces_list = []
    faces_info_list = []

    for photo in images_photos:
        number_faces = len(photo.faces)
        faces = list(photo.faces)
        photo_ids.append(photo.photoId)
        num_faces_list.append(number_faces)
        faces_info_list.append(faces)

    face_info_df = pd.DataFrame({
        'image_id': photo_ids,
        'n_faces': num_faces_list,
        'faces_info': faces_info_list
    })


    meta_file = os.path.join(input_path, "bg_segmentation.pb")

    try:

        meta_info_bytes_info_bytes = open(meta_file, 'rb').read()
        meta_descriptor = meta_vector.PhotoBGSegmentationMessageWrapper()
        meta_descriptor.ParseFromString(meta_info_bytes_info_bytes)

        if meta_descriptor.WhichOneof("versions") == 'v1':
            message_data = meta_descriptor.v1

        images_photos = message_data.photos
        # Prepare lists to collect data
        photo_ids = []
        image_times = []
        scene_orders = []
        image_aspects = []
        image_colors = []
        image_orientations = []
        image_orderInScenes = []
        background_centroids = []
        blob_diameters = []

        # Add safer handling of photo attributes
        for photo in images_photos:
            photo_ids.append(photo.photoId)
            image_times.append(photo.dateTaken)
            scene_orders.append(photo.sceneOrder)
            image_aspects.append(photo.aspectRatio)
            image_colors.append(photo.colorEnum)
            image_orientations.append('landscape' if photo.aspectRatio >= 1 else 'portrait')
            image_orderInScenes.append(photo.orderInScene)
            # Safer handling of optional fields
            background_centroids.append(getattr(photo, 'blobCentroid', None))
            blob_diameters.append(getattr(photo, 'blobDiameter', None))

        additional_image_info_df = pd.DataFrame({
            'image_id': photo_ids,
            'image_time': image_times,
            'scene_order': scene_orders,
            'image_as': image_aspects,
            'image_color': image_colors,
            'image_orientation': image_orientations,
            'image_orderInScene': image_orderInScenes,
            'background_centroid': background_centroids,
            'diameter': blob_diameters
        })
    except Exception as ex:
        print(f"Error reading photo meta info from file: {ex}")


    # Merge the two DataFrames on 'image_id'
    df = pd.merge(face_info_df, additional_image_info_df, on='image_id', how='left')

    process_crop_images(q,df)

    print("done")



#: Columns `face_aware_crop` needs beyond `image_as`, all produced by the
#: gallery read: the face boxes, the saliency blob and its size.
_CROP_INPUTS = ('faces_info', 'background_centroid', 'diameter')


def face_aware_crop(image_info, target_ar, logger=None):
    """A crop of `target_ar` positioned to keep the faces, or None.

    `process_cropping` has always been able to do this -- its general branch
    builds a face mask and hands it to `crop_find` -- but the only caller,
    `process_crop_images`, passes ``box_aspect_ratio=1``, so the frame carries
    a *square* crop and nothing else. Anything wider fell through to
    `customize_box`, which centres the window blind.

    On 53507032's cover box (1.96:1) a portrait keeps 34% of its height, so the
    centred band was y=0.33 to 0.67 -- above the faces. Of the 48 couple
    candidates the centre crop kept the faces in 18; a window of the same
    height, merely repositioned, contains them in 47. The height was never the
    problem, only where it sat.

    None when the inputs are missing or the search fails, so the caller can
    keep its own behaviour rather than lose the placement.
    """
    for column in _CROP_INPUTS:
        try:
            value = image_info[column]
        except (KeyError, IndexError):
            if logger:
                logger.info(f"face-aware crop: no '{column}', keeping the centred crop")
            return None
        if value is None:
            return None

    faces = image_info['faces_info']
    if not isinstance(faces, list):
        faces = list(faces) if faces is not None else []
    # blurLevel -1 is what Face-Recognition writes for a detection its own check
    # judged not a face; on 53840120 both covers had one, in a corner, and the
    # crop moved to keep it.
    faces = [f for f in faces if getattr(f, 'blurLevel', 0) >= 0]

    stand_ins = []
    try:
        stand_ins = _hidden_couple_faces(image_info, faces, float(image_info['image_as']), logger)
    except Exception as exc:  # noqa: BLE001 - same rule as the crop below
        if logger:
            logger.warning(f"face-aware crop: could not place a hidden face "
                           f"({type(exc).__name__}: {exc})")
    faces = faces + stand_ins

    if not faces:
        # No face to aim at; the centred crop is as good a guess as any.
        return None

    try:
        image_ar = float(image_info['image_as'])
        crop = process_cropping(
            image_ar,
            faces,
            image_info['background_centroid'],
            float(image_info['diameter']),
            float(target_ar),
        )
        crop = _faces_first(crop, faces, image_ar, float(target_ar), logger,
                            centre_on_union=bool(stand_ins))
    except Exception as exc:  # noqa: BLE001 - a crop is not worth the album
        if logger:
            logger.warning(f"face-aware crop failed ({type(exc).__name__}: {exc}); "
                           "keeping the centred crop")
        return None

    # Plain floats. `process_cropping` divides numpy ints, so it answers in
    # `np.float64`, and the album doc goes out through `json.dumps` with no
    # `default=` handler in `push_report_msg`. That happens to work --
    # `np.float64` subclasses `float` -- but nothing here should depend on
    # that, and `customize_box` has always returned plain floats.
    return tuple(float(v) for v in crop)


#: COCO keypoint indices of the head: nose, eyes, ears.
_HEAD_KEYPOINTS = range(5)
_MIN_KEYPOINT_SCORE = 0.3
#: A head box's side, as a multiple of the spread of its keypoints. Eyes and
#: ears span about half a head, and the face box the crop is used to is
#: roughly a head.
_HEAD_FROM_KEYPOINTS = 1.6
#: Share of a body box taken as its head when the keypoints are too weak.
_HEAD_SHARE_OF_BODY = 0.25
#: Half-size of the stand-in box put on the saliency centre when no body is
#: found either -- a point the window must contain, not a region to fill.
_BLOB_POINT_HALF = 0.02


def _box(x1, y1, x2, y2):
    """A stand-in face the crop code reads like a detection.

    blurLevel 0 is below ``cover_crop_min_face_blur``, so a stand-in never
    becomes the main face `_faces_first` centres on -- a detected face leads.
    """
    clamp = lambda v: min(max(v, 0.0), 1.0)
    bbox = SimpleNamespace(x1=clamp(x1), y1=clamp(y1), x2=clamp(x2), y2=clamp(y2))
    return SimpleNamespace(bbox=bbox, blurLevel=0.0)


def _couple_ids(image_info):
    ids = set()
    for column in ('bride_id', 'groom_id'):
        try:
            value = image_info[column]
        except (KeyError, IndexError):
            continue
        if value is not None and not pd.isna(value):
            ids.add(value)
    return ids


def _listed(image_info, column):
    try:
        value = image_info[column]
    except (KeyError, IndexError):
        return []
    if value is None or isinstance(value, (str, float)):
        return []
    # Not `hasattr(value, '__iter__')`: the upb protobuf container that
    # `bodies_info` holds lists fine but has no such attribute.
    try:
        return list(value)
    except TypeError:
        return []


def _contains_face(body, faces):
    b = body.bbox
    for f in faces:
        cx, cy = (f.bbox.x1 + f.bbox.x2) / 2, (f.bbox.y1 + f.bbox.y2) / 2
        if b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2:
            return True
    return False


def _head_of(body, image_ar):
    """The head of a body detection, from its keypoints or its top."""
    points = [kp for i, kp in enumerate(getattr(body, 'keypoints', []))
              if i in _HEAD_KEYPOINTS and kp.score >= _MIN_KEYPOINT_SCORE
              and 0.0 <= kp.x <= 1.0 and 0.0 <= kp.y <= 1.0]
    if len(points) >= 2:
        xs, ys = [p.x for p in points], [p.y for p in points]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        # Square in pixels: x is a fraction of the width, y of the height.
        side = _HEAD_FROM_KEYPOINTS * max((max(xs) - min(xs)) * image_ar, max(ys) - min(ys))
        half_x, half_y = side / 2 / image_ar, side / 2
        return _box(cx - half_x, cy - half_y, cx + half_x, cy + half_y)

    b = body.bbox
    return _box(b.x1, b.y1, b.x2, b.y1 + (b.y2 - b.y1) * _HEAD_SHARE_OF_BODY)


def _hidden_couple_faces(image_info, faces, image_ar, logger=None):
    """Stand-in faces for the couple members this photo holds without a face.

    Weddings only -- `bride_id` / `groom_id` are resolved for nothing else. An
    identity can be recognised in a photo whose face the detector missed: on
    49994361's closing photo the groom's hand covered his face, identity 13 was
    placed there with no face box, and the crop centred on the bride alone and
    cut him in half. Every face-led rule here then reads "all faces fit" as
    "the bride fits".

    Each hidden member gets the head of a body that holds no detected face,
    largest first -- a guest cut by the frame edge is smaller than the partner
    beside the one who was found. With no such body, the saliency centre stands
    in: it at least pulls the window towards the subject instead of away.
    """
    hidden = _couple_ids(image_info) & set(_listed(image_info, 'faceless_persons_ids'))
    if not hidden:
        return []

    bodies = [b for b in _listed(image_info, 'bodies_info') if not _contains_face(b, faces)]
    bodies.sort(key=lambda b: (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1), reverse=True)

    stand_ins = [_head_of(body, image_ar) for body in bodies[:len(hidden)]]
    source = 'body'
    if not stand_ins:
        centroid = image_info['background_centroid']
        if centroid is None:
            return []
        stand_ins = [_box(centroid.x - _BLOB_POINT_HALF, centroid.y - _BLOB_POINT_HALF,
                          centroid.x + _BLOB_POINT_HALF, centroid.y + _BLOB_POINT_HALF)]
        source = 'saliency centre'

    if logger:
        logger.info(f"face-aware crop: couple member(s) {sorted(hidden)} in frame without a face; "
                    f"keeping {len(stand_ins)} stand-in from the {source}")
    return stand_ins


def _window_size(image_ar, target_ar):
    """The (w, h) of the largest `target_ar` window in the image, normalised."""
    if image_ar > target_ar:
        return target_ar / image_ar, 1.0
    return 1.0, image_ar / target_ar


def _holds(crop, box, tol=1e-6):
    x, y, w, h = crop
    x1, y1, x2, y2 = box
    return x - tol <= x1 and y - tol <= y1 and x2 <= x + w + tol and y2 <= y + h + tol


def _centred_on(box, w, h):
    """A w x h window centred on `box`, slid back inside the image."""
    x1, y1, x2, y2 = box
    x = min(max((x1 + x2) / 2 - w / 2, 0.0), 1.0 - w)
    y = min(max((y1 + y2) / 2 - h / 2, 0.0), 1.0 - h)
    return x, y, w, h


def _faces_first(crop, faces, image_ar, target_ar, logger=None, centre_on_union=False):
    """Make sure the faces decide the cover crop, not `process_cropping`'s penalties.

    `process_cropping` grows every face box by a face on each side before its
    search, so a large face in a wide box becomes a mask taller than the window
    and every position misses about as much of it; a small face elsewhere then
    decides. 53840120 opened on a portrait cut through the bride's mouth to
    keep a 0.1-wide detection in the corner.

    If one window can hold every face box, the search's crop stands when it
    does hold them, and otherwise the window is centred on them all. If not,
    the window is centred on the main face: the largest one at least
    ``cover_crop_min_face_blur`` sharp, so a large blurred face in the
    foreground cannot lead. With no face that sharp the search's crop stands.

    ``centre_on_union`` centres on the faces even when the search's crop
    already holds them. It is set when a stand-in for a hidden face is among
    them: the search does keep that head in, but on 49994361's closing photo it
    did so by sliding as far towards it as it could and cut the bride's hair at
    the other edge.
    """
    boxes = [(f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2) for f in faces]
    w, h = _window_size(image_ar, target_ar)
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes))

    if union[2] - union[0] <= w and union[3] - union[1] <= h:
        if not centre_on_union and all(_holds(crop, b) for b in boxes):
            return crop
        return _centred_on(union, w, h)

    min_blur = CONFIGS['cover_crop_min_face_blur']
    sharp = [(b, f) for b, f in zip(boxes, faces)
             if getattr(f, 'blurLevel', min_blur) >= min_blur]
    if not sharp:
        return crop

    main, face = max(sharp, key=lambda bf: (bf[0][2] - bf[0][0]) * (bf[0][3] - bf[0][1]))
    if logger:
        logger.info(f"cover crop: {len(faces)} faces do not fit one window; centring on the "
                    f"main face (blurLevel {getattr(face, 'blurLevel', 'n/a')})")
    return _centred_on(main, w, h)

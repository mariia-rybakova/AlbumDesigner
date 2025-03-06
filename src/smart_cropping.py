import cv2
import numpy as np


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

            x1 = int(bbox.x1)
            y1 = int(bbox.y1)
            x2 = int(bbox.x2)
            y2 = int(bbox.y2)
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
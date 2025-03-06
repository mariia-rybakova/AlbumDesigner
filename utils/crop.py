import cv2
import numpy as np
from PIL import Image


def cropWeight(tl, br, foregroundMask=None, faceMask=None, aspectRatio=1, center_weight=200, face_weight=150):
    tl = np.array(tl).astype(np.int32)

    imageShape = br
    if (imageShape[0] - tl[0]) * aspectRatio < (imageShape[1] - tl[1]):
        height = int(imageShape[0] - tl[0])
        width = int((imageShape[0] - tl[0]) * aspectRatio)
    else:
        height = int((imageShape[1] - tl[1]) / aspectRatio)
        width = int((imageShape[1] - tl[1]))

    sizeRatio = width * height / (imageShape[0] * imageShape[1])

    center = np.array((tl[0] + width / 2, tl[1] + height / 2))
    o_center = np.array(foregroundMask.shape) / 2

    fl_tl = np.floor(tl).astype(int)

    croppedForeground = foregroundMask[fl_tl[0]:fl_tl[0] + height, fl_tl[1]:fl_tl[1] + width]
    fl_weight = np.abs(np.sum(foregroundMask) - np.sum(croppedForeground))
    if faceMask is not None:
        croppedFace = faceMask[fl_tl[0]:fl_tl[0] + height, fl_tl[1]:fl_tl[1] + width]
        fl_weight += np.abs(np.sum(faceMask) - np.sum(croppedFace)) * face_weight

    fl_weight += np.linalg.norm(o_center - center) * center_weight

    return fl_weight, width, height


def cropStep(xRange, yRange, fxRange, fyRange, foregroundMask, faceMask=None, aspectRatio=1, steps=3):
    x = np.linspace(xRange[0], xRange[1], steps + 1)[:steps]
    y = np.linspace(yRange[0], yRange[1], steps + 1)[:steps]

    fx = np.linspace(fxRange[0], fxRange[1], steps + 1)[:steps]
    fy = np.linspace(fyRange[0], fyRange[1], steps + 1)[:steps]

    cr_weights = np.zeros((steps, steps))
    w = np.zeros((steps, steps))
    h = np.zeros((steps, steps))

    for xi in range(len(x)):
        for yi in range(len(y)):
            temp_cr_weight = 1e16
            for fxi in range(len(fx)):
                for fyi in range(len(fy)):
                    temp_weight, temp_w, temp_h = cropWeight([x[xi], y[yi]], [fx[fxi], fy[fyi]], foregroundMask,
                                                             faceMask, aspectRatio)
                    if temp_weight < temp_cr_weight:
                        temp_cr_weight = temp_weight
                        w[xi, yi] = temp_w
                        h[xi, yi] = temp_h
            cr_weights[xi, yi] = temp_cr_weight

    m = np.where(cr_weights == np.min(cr_weights))
    if len(m[0]) == 1:
        return m, x, y, w[m[0].item(), m[1].item()], h[m[0].item(), m[1].item()]
    else:
        return m, x, y, w[m[0][0].item(), m[0][1].item()], h[m[0][0].item(), m[0][1].item()]


def crop_find(foregroundMask, faceMask=None, aspectRatio=1, steps=3):
    s_min = [0, 0]
    s_max = [foregroundMask.shape[0], foregroundMask.shape[1]]

    im_shape = [foregroundMask.shape[0], foregroundMask.shape[1]]

    while s_max[0] - s_min[0] > 9 and s_max[1] - s_min[1] > 9:
        m, x, y, w, h = cropStep([s_min[0], s_max[0]], [s_min[1], s_max[1]], [2 * im_shape[0] // 3, im_shape[0]],
                                 [2 * im_shape[1] // 3, im_shape[1]], foregroundMask, faceMask, aspectRatio, steps)
        if len(m[0]) > 1:
            m = (m[0][0], m[1][0])
        s_min[0] = x[m[0]].item()
        if m[0] < len(x) - 1:
            s_max[0] = x[m[0] + 1].item()
        s_min[1] = y[m[1]].item()
        if m[1] < len(y) - 1:
            s_max[1] = y[m[1] + 1].item()
        m

    return s_min, s_max, int(w), int(h)


def smart_cropping(ar, faces, centroid, diameter,box_aspect_ratio, min_dim=1000, face_extenssion=2):
    if ar > 1:
        mask = np.zeros((min_dim, int(ar * min_dim)), dtype=np.uint8)
    else:
        mask = np.zeros((int(min_dim / (ar * 1.0)), min_dim), dtype=np.uint8)


    mask = cv2.circle(mask, (int(centroid.y * mask.shape[1]), int(centroid.x * mask.shape[0])), int(diameter / 2 * mask.shape[0]), 255, -1)

    if len(faces) != 0:
        if not isinstance(faces, list):
            faces = list(faces)

        faces = faces[0]
        for face in faces:
            face.bbox.x1 = int(face.bbox.x1 * mask.shape[1])
            face.bbox.y1 = int(face.bbox.y1 * mask.shape[0])
            face.bbox.x2 = int(face.bbox.x2 * mask.shape[1])
            face.bbox.y2 = int(face.bbox.y2 * mask.shape[0])



    face_mask = None
    if len(faces) > 0:
        face_mask = np.zeros_like(mask, dtype=np.uint8)
        if isinstance(faces, list):
            for face in faces:
                bbox = face.bbox
                x1,y1,x2,y2 = np.int32(bbox.x1), np.int32(bbox.y1), np.int32(bbox.x2), np.int32(bbox.y2)
                bbox_h = (y2 - y1) * face_extenssion
                bbox_w = (x2 - x1) * face_extenssion

                x1 = int(max(0, x1 - bbox_w / 2))
                y1 = int(max(0, y1 - bbox_h / 2))
                x2 = int(min(mask.shape[1], x2 + bbox_w / 2))
                y2 = int(min(mask.shape[0], y2 + bbox_h / 2))

                single_face_mask = mask[y1:y2, x1:x2]
                if single_face_mask.sum() > 0:
                    face_mask[y1:y2, x1:x2] = 255


    s_min, s_max, w, h = crop_find(mask, faceMask=face_mask, aspectRatio=box_aspect_ratio, steps=4)

    return s_min[0] / mask.shape[0], s_min[1] / mask.shape[1], w / mask.shape[1], h / mask.shape[0]


if __name__ == "__main__":
    class BBox(object):
        def __init__(self, x1, y1, x2, y2):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

    class Face:
        def __init__(self, x1, y1, x2, y2):
            self.bbox = BBox(x1, y1, x2, y2)

    class Centroid:
        def __init__(self, x1, y1):
            self.x = x1
            self.y = y1

    image_id = 9850153891
    ar = 1.0
    faces  = [[Face(0.303076923,0.564615369,0.392307699,0.696923077)]]
    centroid  = Centroid(0.799326658,0.321599692)
    diameter = 0.142827

    cropped_x,cropped_y,cropped_w,cropped_h = smart_cropping(ar, faces, centroid, diameter, min_dim=1000, face_extenssion=2)

    image = Image.open(r'/AlbumDesignQueue/dataset\40850524\9850153891.jpg').convert('RGB')

    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    s_min = np.zeros(2, dtype=int)
    x = int(cropped_x * np_img.shape[0])
    y = int(cropped_y * np_img.shape[1])
    h = int(cropped_h * np_img.shape[0])
    w = int(cropped_w * np_img.shape[1])

    crop = np.array(image)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cropped_image = crop[x:x + h,y:y + w]
    # Resize the cropped image
    img = Image.fromarray(cropped_image)
    img.show()
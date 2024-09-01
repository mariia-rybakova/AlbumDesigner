import cv2
import numpy as np


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


def smart_cropping(ar, faces, centroid, diameter, min_dim=1000, face_extenssion=2):
    if ar > 1:
        mask = np.zeros((min_dim, int(ar * min_dim)), dtype=np.uint8)
    else:
        mask = np.zeros((int(min_dim / (ar * 1.0)), min_dim), dtype=np.uint8)

    mask = cv2.circle(mask, (int(centroid[1] * mask.shape[1]), int(centroid[0] * mask.shape[0])),
                      int(diameter / 2 * mask.shape[0]), 255, -1)

    for face in faces:
        face['bbox'][0] = int(face['bbox'][0] * mask.shape[1])
        face['bbox'][1] = int(face['bbox'][1] * mask.shape[0])
        face['bbox'][2] = int(face['bbox'][2] * mask.shape[1])
        face['bbox'][3] = int(face['bbox'][3] * mask.shape[0])

    face_mask = None
    if len(faces) > 0:
        face_mask = np.zeros_like(mask, dtype=np.uint8)
        for face in faces:
            bbox = face['bbox'].astype(np.int32)
            bbox_h = (bbox[3] - bbox[1]) * face_extenssion
            bbox_w = (bbox[2] - bbox[0]) * face_extenssion

            bbox[0] = int(max(0, bbox[0] - bbox_w / 2))
            bbox[1] = int(max(0, bbox[1] - bbox_h / 2))
            bbox[2] = int(min(mask.shape[1], bbox[2] + bbox_w / 2))
            bbox[3] = int(min(mask.shape[0], bbox[3] + bbox_h / 2))

            single_face_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if single_face_mask.sum() > 0:
                face_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    s_min, s_max, w, h = crop_find(mask, faceMask=face_mask, aspectRatio=1, steps=4)

    return s_min[0] / mask.shape[0], s_min[1] / mask.shape[1], w / mask.shape[1], h / mask.shape[0]

#
# # files = glob(r'C:\temp\wed apr 24\38256105\*.jpg')
# files = glob(r'C:\temp\face_issue\*.jpg')
#
# # files = glob(r'C:\temp\1746253403_S2_O82.jpg')
# # files = glob(r'C:\temp\croping images\*.jpg')
# import matplotlib.pyplot as plt
#
# remover = Remover() # default setting
# app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
# app.prepare(ctx_id=0, det_size=(1024, 1024))
#
# face_extenssion = 2
#
# img = ins_get_image('t1')
#
# all_faces= []
# all_centroids=[]
# all_diameters=[]
# all_ars = []
#
# for idx,file in enumerate(tqdm(files)):
#
#
#     # img = Image.open(r'C:\Users\ZivRotman\Downloads\423328036_18390166600069932_759402819709351851_n.jpg').convert('RGB')
#     # img = Image.open(r'C:\temp\dec23 weds\batch3\29865519\8048999467.jpg').convert('RGB')
#     img = Image.open(file).convert('RGB')
#
#     img_r = img.resize((img.size[0]//3,img.size[1]//3))
#     # img_r=img
#
#     # img = ins_get_image('t1')
#     # faces = app.get(img)
#
#     out = remover.process(img_r) # default setting - transparent background
#
#
#     np_img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
#     faces = app.get(np_img)
#
#     for face in faces:
#         face['bbox'][0] = face['bbox'][0]/np_img.shape[1]
#         face['bbox'][1] = face['bbox'][1]/np_img.shape[0]
#         face['bbox'][2] = face['bbox'][2]/np_img.shape[1]
#         face['bbox'][3] = face['bbox'][3]/np_img.shape[0]
#
#
#     all_faces.append(faces)
#
#
#
#     np_out = np.array(out)
#     mask = np.squeeze(np_out[:,:,3])>0
#     mask_o = np.copy(mask)
#
#     mask_r = cv2.resize(mask.astype(np.uint8),(np_img.shape[1],np_img.shape[0]))
#     mask = mask_r
#
#     label_img = label(mask)
#     regions = sorted(regionprops(label_img),key=lambda r: r.area, reverse=True)
#
#     all_centroids.append([regions[0].centroid[0]/mask.shape[0],regions[0].centroid[1]/mask.shape[1]])
#     all_diameters.append(regions[0].equivalent_diameter_area/mask.shape[0])
#
#
#
#     all_ars.append(np_img.shape[1]/np_img.shape[0])
#
#


# for idx,file in enumerate(tqdm(files)):
#
#     smin0, smin1,w,h = cropImage(all_ars[idx],all_faces[idx],all_centroids[idx],all_diameters[idx],min_dim=1000)
#
#     image = Image.open(file).convert('RGB')
#     np_img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
#
#
#     s_min = np.zeros(2,dtype=int)
#     s_min[0] = int(smin0*np_img.shape[0])
#     s_min[1] = int(smin1*np_img.shape[1])
#     h = int(h*np_img.shape[0])
#     w = int(w*np_img.shape[1])
#
#
#     crop = np.array(image)
#     crop = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
#     crop = crop[s_min[0]:s_min[0]+h,s_min[1]:s_min[1]+w]
#
#     # crop_resized = cv2.resize(crop,(512,512))
#
#     os.makedirs('crops', exist_ok=True)
#     cv2.imwrite('crops/'+os.path.split(file)[1],crop)
#     # fileonly,ext = os.path.splitext(os.path.split(file)[1])
#     # out.save('crops/'+fileonly+'.png')
#     crop

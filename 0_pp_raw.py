import numpy as np
from matplotlib import pyplot as plt
import cv2
fpath='/home/cisir4/anaconda3/resources/ocsource/batch1/raw_rest/raw_NI_DG_Bf.npz'

import math

def getDetBoxes_core(image_x,textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    import copy

    linkmap = linkmap.copy()
    textmap = textmap.copy()
    cross_check=copy.deepcopy(textmap)
    img_h, img_w = textmap.shape

    # pts=np.array([[418, 216.], [507, 216.],
    #               [507, 250.], [418, 250.]],
    #              np.int32)

    # pts = pts.reshape((-1, 1, 2))
    # img_box=cv2.polylines(cross_check, [pts], True,1,thickness=2)
    # plt.imshow(img_box)
    # plt.show()

    # plt.subplot(2, 2, 1)
    # plt.imshow(linkmap)

    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    """ labeling method """

    # Creating kernel
    kernel = np.ones((10, 10), np.uint8)

    # Using cv2.erode() method
    image = cv2.erode(textmap.copy(), kernel)

    # plt.subplot(2, 2, 2)
    # plt.imshow(image)

    # Adjust these two value to get a good bbox
    dil_hor=20
    dil_ver=10

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(dil_hor, dil_ver))

    dilated = cv2.dilate(image, kernel)

    # plt.subplot(2, 2, 3)
    # plt.imshow(dilated)

    # plt.show()

    low_dilate=0.3
    ret, text_score_dilate = cv2.threshold(dilated, low_dilate, 1, 0)

    # plt.subplot(2, 2, 4)
    # plt.imshow(text_score_dilate)

    # plt.show()

    # ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(text_score_dilate )
    # plt.subplot(2, 2, 4)
    # plt.imshow(text_score)


    # plt.show()

    # text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_dilate.astype(np.uint8),4,cv2.CV_32S)

    # Always omit first index as in refer to the cp of big image
    # plt.imshow(text_score_dilate)
    # plt.scatter(centroids[1:,0], centroids[1:,1], c ="blue",s=5,marker='^')
    # plt.show()
    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 1000: continue

        # thresholding
        # if np.max(textmap[labels==k]) < text_threshold: continue
        print(f'The box size is: {size}')
        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        # segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    # pts=np.array([[418, 216.], [507, 216.],
    #               [507, 250.], [418, 250.]],
    #               np.int32)

    # pts = pts.reshape((-1, 1, 2))
    # img_box=cv2.polylines(cross_check, [pts], True,1,thickness=2)
    # plt.imshow(img_box)
    # plt.show()
    r=1
    return det, labels, mapper

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


data = np.load(fpath)
textmap=data['score_text']
linkmap=data['score_linkt']
image=data['image']
text_threshold=0.2
link_threshold=0.2
low_text=0.2
boxes, labels, mapper = getDetBoxes_core(image,textmap, linkmap,
                                         text_threshold, link_threshold,
                                         low_text)

ratio_h =  1.1953125
ratio_w =1.1953125
target_ratio = 0.8366013071895425

boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)


img = np.array(image)
for dbox in boxes:

    pts = dbox.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], True,1,thickness=2)


plt.imshow(img)
plt.show()

h=1
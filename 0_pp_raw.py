import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from os import path as osp
import os
import statistics
# from glob import glob
from tqdm import tqdm
def getDetBoxes_core(textmap, char_s,dil_hor,dil_ver,sep_vertH,sep_vertW):

    # prepare data

    textmap = textmap.copy()

    img_h, img_w = textmap.shape


    """ labeling method """

    # Creating kernel
    # Try to seperate between level
    # sep_vert=10
    kernel = np.ones((sep_vertH, sep_vertW), np.uint8)

    image = cv2.erode(textmap.copy(), kernel)



    # Adjust these two value to get a good bbox
    # dil_hor=20
    # dil_ver=10
    # sep_vert=10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(dil_hor, dil_ver))

    dilated = cv2.dilate(image, kernel)


    low_dilate=0.3
    ret, text_score_dilate = cv2.threshold(dilated, low_dilate, 1, 0)


    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_dilate.astype(np.uint8),4,cv2.CV_32S)




    det = []
    mapper = []
    cent=[]
    bb_ls=[]
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < char_s: continue

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
        # mcent=centroids[k]
        # cent.append(mcent)
        # det.append(box)
        # mapper.append(k)
        xxis=0
        yxis=1
        bheight=box[3,yxis]-box[0,yxis] # between y axis
        bwidth=box[2,xxis]-box[0,xxis] # between x axis
        # if diff_height != h:
        #     print('the height are same')
        #
        # if diff_width != w:
        #     print('the weight are same')
        bb_ls.append(dict(bb=box,
                          cent=centroids[k],
                          mapper=k,
                          height=bheight,
                          width=bwidth))
        t=1



    # Lets do further checking.
    """
    
    # Also, convert the distance to a ratio 
    # Get the box centroid,stats.
    
    # get point(coordinate) that almost vertical to each other.
    # 
    """
    nlim=0
    CentValMin=300
    CentValMax=600
    # Check how many bbox availabe
    nbox=len(bb_ls)

    ## I think it is best first remove unnecesary bbox.
    # gg=stats[mapper, cv2.CC_STAT_AREA].tolist()

    if nbox >=4:
        nbbls=[]

        for dcent in bb_ls:
            t=dcent['cent'][1]
            if (CentValMin<dcent['cent'][0]<CentValMax) &(60<dcent['cent'][1]<700):
                nbbls.append(dcent)
                j=1
        # pot_val.append(dict(bboxes=lst_int))
        s=1
        bb_ls=nbbls


    nbox=len(bb_ls)
    if nbox==3:

        det = list(map (lambda x:x['bb'],bb_ls))
        bheight=np.array(list(map (lambda x:x['height'],bb_ls)))
        bwidth=np.array(list(map (lambda x:x['width'],bb_ls)))
        arr=np.array(det)
        # I want to know if all have same size, this is identify if there is a single character

        dmapper= [ item['mapper'] for item in  bb_ls ]

        # nvar_area=statistics.stdev(stats[dmapper, cv2.CC_STAT_AREA].tolist())


        nvar_height=np.std(bheight)
        nvar_width=np.std(bwidth)

        # g=stats[dmapper, cv2.CC_STAT_WIDTH].tolist()
        h=1
        if nvar_width>10:
            """
            This apply for 
          
            """
            # High chances there is a bbox size mismatch.
            # For time being (although i expect thing to be modified later, we maximise the bbox base on the largest bbox as our box reference
            mxwidtht=np.amax(bwidth)
            idx_ref = np.where(bwidth == np.amax(bwidth))[0].tolist()[0]
            xxis=0
            yxis=1
            xleft_ref=bb_ls[idx_ref ]['bb'][0,xxis]
            xright_ref=bb_ls[idx_ref ]['bb'][1,xxis]
            for idx in range(len(bb_ls)):
                dheight=bb_ls[idx]['width']
                diff_width=(mxwidtht-dheight)/2
                if diff_width>15:
                    # Adjust the bbox so that it widen in the up and bottom direction (Y axis)
                    # cval=bb_ls[idx]['bb'][[0],0]
                    xxis=0
                    yxis=1


                    bb_ls[idx]['bb'][[0,3],xxis]=xleft_ref
                    bb_ls[idx]['bb'][[1,2],xxis]=xright_ref



        if nvar_height>5:
            # bbs_height=stats[dmapper, cv2.CC_STAT_HEIGHT].tolist()
            mxheihgt=np.max(bheight)
            # mheight=bbs_height.index(max(bbs_height))

            for idx in range(len(bb_ls)):
                dheight=bb_ls[idx]['height']
                diff_height=(mxheihgt-dheight)/2
                if diff_height>5:
                    # Adjust the bbox so that it widen in the up and bottom direction (Y axis)
                    # cval=bb_ls[idx]['bb'][[0],0]
                    xxis=0
                    yxis=1

                    ya=bb_ls[idx]['bb'][0,yxis] # Top
                    yd=bb_ls[idx]['bb'][3,yxis] # Bottom
                    bb_ls[idx]['bb'][[0,1],yxis]=ya-diff_height
                    bb_ls[idx]['bb'][[2,3],yxis]=yd+diff_height
                    bb_ls[idx]['height']=bb_ls[idx]['bb'][3,yxis]-bb_ls[idx]['bb'][0,yxis]


    elif nbox==2:
        # At this stage, we always assume, the second and third row always combined
        # First, find the horizontal midline

        # I am not in favor of this approach, usually, set higher sep_vertW to 16 should do the trick especially for 'raw_N3_DA_GJ_4'
        idx=1
        ya=bb_ls[idx]['bb'][0,yxis] # Top
        yd=bb_ls[idx]['bb'][3,yxis] # Bottom
        ymidline=ya+((yd-ya)/2)  # Our horizontal midline coordinate
        xa=bb_ls[idx]['bb'][0,0]
        xb=bb_ls[idx]['bb'][1,0]
        # Coordinate for midle box
        bb_mid=np.array([bb_ls[idx]['bb'][0,:],
                         bb_ls[idx]['bb'][1,:],
                         [xb,ymidline],
                         [xa,ymidline]])
        dic_mid=dict(bb=bb_mid,
                     mapper='NA',
                     height=ymidline-bb_ls[idx]['bb'][0,1],
                     width=xb-xa)

        bb_bot=np.array([[xa,ymidline],
                        [xb,ymidline],
                        bb_ls[idx]['bb'][2,:],
                        bb_ls[idx]['bb'][3,:]])

        dic_bot=dict(bb=bb_bot,
                     mapper='NA',
                     height=bb_ls[idx]['bb'][3,1]-ymidline,
                     width=xb-xa)
        bb_ls=[bb_ls[0],dic_mid,dic_bot]
        w=1

    ### SOMETIME, THERE IS another stray box that stay at the top of upper most character, need to find a way to drop this
    nbox=len(bb_ls)
    if nbox >=4:
        print('Need to remove the extra box, which is located at the upper most region')
    ff=1
    h=1
    # return det, labels, mapper,text_score_dilate
    return bb_ls,text_score_dilate

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys



def process_img(image,textmap,linkmap,spath):
    dil_hor=30
    dil_ver=10
    sep_vertW=15
    sep_vertH=10
    char_s=400

    bb_dic,img_dil=getDetBoxes_core(textmap, char_s,dil_hor,dil_ver,
                                                   sep_vertH,sep_vertW)

    ratio_h =  1.1953125
    ratio_w =1.1953125
    target_ratio = 0.8366013071895425
    # plt.imshow(linkmap)
    # plt.show()
    boxes=list(map (lambda x:x['bb'],bb_dic))
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)


    img = np.array(image)
    for dbox in boxes:

        pts = dbox.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts], True,1,thickness=5)



    plt.figure(figsize = (16,8))
    plt.subplot(1, 2, 1), plt.imshow(img, 'gray')
    plt.subplot(1, 2, 2), plt.imshow(img_dil, 'gray')
    plt.tight_layout()
    plt.show()
    # plt.savefig(spath) # To save figure
    # plt.close('all')
    h=1


batch_idx=2
# path_all = glob ( f"/home/cisir4/anaconda3/resources/ocsource/batch{batch_idx}/raw_rest/*.npz" )


# fname='raw_N3_DA_GJ_4'
fname='raw_N3_DA_Im_1'
path_all=[f"/home/cisir4/anaconda3/resources/ocsource/batch{batch_idx}/raw_rest/{fname}.npz" ]
# all_path=d
for dpath in tqdm(path_all):
    # fpath='/home/cisir4/anaconda3/resources/ocsource/batch1/raw_rest/raw_NI_DG_Bf.npz'
    data = np.load(dpath)
    textmap=data['score_text']
    linkmap=data['score_linkt']
    image=data['image']



    rdir,fname=osp.split(dpath)
    droot,_=osp.split(rdir)
    droot=osp.join(droot,'dilation_approach')
    if not osp.isdir(droot):
        os.mkdir(droot)

    dname=fname.split('.npz')[0]
    spath=osp.join(droot,f'{dname}.jpg')

    process_img(image,textmap,linkmap,spath)


    # cv2.imwrite(spath, img)
    # rom PIL import Image
    # img = Image.fromarray(img)

    # print(tesserocr.image_to_text(img))

    # np_img = np.array(img)
    # cv2.imshow('plate',  np_img)

    j=1

print('COMPLETE')
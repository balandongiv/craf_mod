from os import path as osp
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import itertools as it
root='/home/cisir4/anaconda3/resources/ocsource/batch1/craft'
# fname='res_N0_AG_EL' # Ideal image

fname='res_N0_AG_ES' # Example two box at the middle character.
# fname='res_N2_AB_MG' # Example two box at the bottom character.
# fname='res_N2_AG_H9' # Example two box at the middle but with uneven box.
# fname='res_N3_CE_I2' # Example two box at the bottom but with uneven box.
# fname='res_N1_DG_Am' # Example two box at the bottom but with uneven box.
# fname='res_N3_CE_I2' # Example two box at the bottom but with uneven box.

# fname='res_N0_AG_EX' # Example two box at the bottom but with uneven box.

# fname='res_N2_AG_Cd' # Example two box at the middle and bottom character

# fname='res_N1_DG_An' # Example two box at the middle and bottom character
# fname='res_N1_DG_BJ' # Example missing left middle character


# fname='res_N2_DG_Bf' # Example missing left middle character
# fname='res_N1_DG_AS' # Example missing left middle character


# fname='res_N0_CE_A3_1'
# fname='res_71_AH_Bv_1'
# TO DO LAST
# fname='res_N2_DG_Aj' # Example missing half middle and both bottom bbox (KIV)

# fname='res_N2_AG_Cd' # Example missing half middle and both bottom bbox (KIV)
# fname='res_N1_DG_Bj' # Example missing half middle and both bottom bbox (KIV)

img=osp.join(root,f'{fname}.jpg')
bb=osp.join(root,f'{fname}.txt')
# Load image, grayscale, median blur, sharpen image
image = cv2.imread(img)
plt.imshow(image)
plt.show()
nlim=image.shape[0]/2
with open(bb) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

# TODO
"""
Issue croping the RES_N0_AG_EX 
"""
# CentValMin=1000
# CentValMax=1200
CentValMin=nlim-200
CentValMax=nlim+130



# First get only box at centre
pot_val=[]
for dline in lines:
    lst_int = [int(x) for x in dline.split(",")]
    if CentValMin<lst_int[0]<CentValMax:
        pot_val.append(dict(bboxes=lst_int))


def nchars_determination(dpots):
    # Lets get the rectangle size to determine whether the box contain single or double characters

    """
    (min_x,min_y)......................(max_x,min_y)
        .                                   .
        .                                   .
    (min_x,max_y)......................(max_x,max_y)
    """

    min_x=min(dpots['bboxes'][0],dpots['bboxes'][6])
    min_y=min(dpots['bboxes'][1],dpots['bboxes'][3])
    max_x=min(dpots['bboxes'][2],dpots['bboxes'][4])
    max_y=min(dpots['bboxes'][7],dpots['bboxes'][5])
    box_dim_coor=[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]

    dst_img = image[min_y:max_y, min_x:max_x]
    plt.imshow(dst_img)
    plt.show()

    #
    bwidht=abs(min_x-max_x)
    bheight=abs(min_y-max_y)

    charsHeight=110 # Typical height of a bb with two characters
    charsWidth=160  # Typical widht of a bb with two characters
    """
    'res_N1_DG_BJ'   (special case single 1>> 1: w:54 h:77 ; B >> 1: w:80 h:107
    'res_N0_AG_EL'      w:188. h:118
    'res_N0_CE_A3_1'    w:180. h:122
    'res_71_AH_Bv_1'    w:168. h:130
    'res_N0_AG_ES'      N0:w:180. h:119
    """


    if bwidht>charsWidth:
        nchar=2
    else:
        nchar=1

    if (nchar==1) & (bheight<charsHeight):
        vchar=1
    elif (nchar==2) & (bheight<charsHeight):
        vchar=1
    else:
        vchar=3

    dpots.update(dict(nchar=nchar,
                      vchar=vchar,
                      box_dim_coor=box_dim_coor,
                      bheight=bheight,
                      bwidht=bwidht))

    return dpots
pot_val=[nchars_determination (dpots) for dpots in pot_val]


def get_angle(x1,y1,x2,y2):
    return math.degrees(math.atan2(y2-y1, x2-x1))

def combine_modular_bboxes (dpots):
    # Next, if there are more than 3 bboxes, high chances there is level that contain single character at each boxes

    gud_bbox=[]
    single_bbox=[]
    for dpot in dpots:
        if dpot['nchar']>=2:
            gud_bbox.append(dpot)
        else:
            # do something
            single_bbox.append(dpot)
    rr=list(set(it.combinations(range(len(single_bbox)), 2)))
    # rr=[rr[3]]
    nval=[]
    for p0,p1 in rr:
        x1=single_bbox[p0]['bboxes'][0]
        x2=single_bbox[p1]['bboxes'][0]


        # Always ensure the pivot point is at extreme left
        if x1>x2:
            # If x1 is greaterm we need to swap x1 to x2 in order to get left pivot


            x1a=single_bbox[p1]['bboxes'][0]
            y1a=single_bbox[p1]['bboxes'][1]
            x1d=single_bbox[p1]['bboxes'][6]
            y1d=single_bbox[p1]['bboxes'][7]

            x2b=single_bbox[p0]['bboxes'][2]
            y2b=single_bbox[p0]['bboxes'][3]
            x2c=single_bbox[p0]['bboxes'][4]
            y2c=single_bbox[p0]['bboxes'][5]

            min_x=max(x1a,x1d)
            min_y=max(y1a,y2b)
            max_x=max(x2c,x2b)
            max_y=max(y1d,y2c)
            box_dim_coor=[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
            plt.imshow(image)
            # # plt.scatter([min_x,max_x,
            # #              min_x,max_x], [min_y,min_y,
            # #                             max_y,max_y], c ="blue")
            plt.scatter([x1a], [y1a], c ="blue",marker='s')
            plt.scatter([x1d], [y1d], c ="red",marker='s')

            plt.scatter([x2b], [y2b], c ="black",marker='^')
            plt.scatter([x2c], [y2c], c ="green",marker='v')
            # plt.scatter([x2], [y2], c ="black")
            plt.show()
            # Xmax=
            h=1
            angl=abs(get_angle(x1a,y1a,x2b,y2b))
        else:
            x1a=single_bbox[p0]['bboxes'][0]
            y1a=single_bbox[p0]['bboxes'][1]

            x1d=single_bbox[p0]['bboxes'][6]
            y1d=single_bbox[p0]['bboxes'][7]


            x2b=single_bbox[p1]['bboxes'][2]
            y2b=single_bbox[p1]['bboxes'][3]
            x2c=single_bbox[p1]['bboxes'][4]
            y2c=single_bbox[p1]['bboxes'][5]
            min_x=max(x1a,x1d)
            min_y=max(y1a,y2b)
            max_x=max(x2c,x2b)
            max_y=max(y1d,y2c)
            box_dim_coor=[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]

            plt.imshow(image)
            # plt.scatter([min_x,max_x,
            #              min_x,max_x], [min_y,min_y,
            #                             max_y,max_y], c ="blue")
            plt.scatter([x1a], [y1a], c ="blue",marker='s')
            plt.scatter([x1d], [y1d], c ="red",marker='s')

            plt.scatter([x2b], [y2b], c ="black",marker='^')
            plt.scatter([x2c], [y2c], c ="green",marker='v')
            plt.show()
            # Xmax=
            h=1

            angl=abs(get_angle(x1a,y1a,x2b,y2b))

        tt=1
        if angl<12:
            """
            res_N3_CE_I2 require more than 5, lets try 8
            
            res_N1_DG_An
            
            RES_N0_AG_EX require 10
            TODO
            Issue cropping RES_N0_AG_EX, N1_DG_Am
            """
            dst_img = image[min_y:max_y, min_x:max_x]
            plt.imshow(dst_img)
            plt.show()
            jjjk=   dict(nchar=2,
                         box_dim_coor=box_dim_coor,
                         bheight=abs(max_y-max_y),
                         bwidht=(max_x-max_x))
            nval.append(jjjk)
        j=1
        # Get the possible box dimension
    h=1
def disintegrate_bbox(dpots):
    h=1
# combine_modular_bboxes (pot_val)

disintegrate_bbox(pot_val)
y=1
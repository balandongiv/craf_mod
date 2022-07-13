from os import path as osp
root='/home/cisir4/anaconda3/resources/ocsource/batch1/craft'
# fname='res_N0_AG_EL' # Ideal image

fname='res_N0_AG_ES' # Example two box at the middle character
img=osp.join(root,f'{fname}.jpg')
bb=osp.join(root,f'{fname}.txt')


with open(bb) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]


CentValMin=1000
CentValMax=1200

charsHeight=550 # Typical height of a bb with two characters
charsWidth=120  # Typical widht of a bb with two characters

# First get only box at centre
pot_val=[]
for dline in lines:
    lst_int = [int(x) for x in dline.split(",")]
    if CentValMin<lst_int[0]<CentValMax:
        pot_val.append(dict(bboxes=lst_int))

# Lets have a
#         min_x = min(bbox[0:-1:2])
#         min_y = min(bbox[1:-1:2])
#         max_x = max(bbox[0:-1:2])
#         max_y = max(bbox[1:-1:2])
#         box = [
#             min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
#         ]

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

    #
    blenght=abs(min_x-min_y)
    bheight=abs(min_y-max_y)

    if blenght<charsWidth:
        nchar=1
    else:
        nchar=2

    dpots.update(dict(nchar=nchar,
                      box_dim_coor=box_dim_coor,
                      blenght=blenght,
                      bheight=bheight))

    return dpots
pot_val=[nchars_determination (dpots) for dpots in pot_val]

# Next, if there are more than 3 bboxes, high chances there is level that contain single character at each boxes

# def combine_modular_bboxes (dpots):

# nstat=[combine_modular_bboxes (dpots) for dpots in pot_val]
y=1
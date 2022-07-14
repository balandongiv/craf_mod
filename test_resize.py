from PIL import Image

image = Image.open('/home/cisir4/anaconda3/resources/ocsource/batch1/test_resize/N2_KG_K2.jpg')

new_image = image.resize((1024, 1024))

new_image.save('/home/cisir4/anaconda3/resources/ocsource/batch1/test_resize/N2_KG_K2_resize.jpg')
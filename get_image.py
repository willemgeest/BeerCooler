import urllib.request
from PIL import Image

def get_image_IPcamera(IPv4_adress,
                       img_location = 'latest_picture/latest_camera_photo.jpg'):
    picture_url = IPv4_adress + '/photo.jpg'  # get link of picture
    urllib.request.urlretrieve(picture_url, img_location)  # download picture to location
    image = Image.open(img_location)  # open image
    image = image.rotate(270, expand=True)  # rotate image
    return image

def resize_image(image, max_width = 500, max_heigth = 600):
    w_factor = image.size[0] / max_width
    h_factor = image.size[1] / max_heigth
    if w_factor > h_factor:
        image = image.resize((int(image.size[0] / w_factor), int(image.size[1] / w_factor)), Image.ANTIALIAS)
    else:
        image = image.resize((int(image.size[0] / h_factor), int(image.size[1] / h_factor)), Image.ANTIALIAS)
    return image


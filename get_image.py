import urllib.request
from PIL import Image

def get_image(IPv4_adress = 'http://192.168.178.108:8080',
              img_location = 'latest_picture/latest_camera_photo.jpg'):
    picture_url = IPv4_adress + '/photo.jpg'  # get link of picture
    urllib.request.urlretrieve(picture_url, img_location)  # download picture to location
    image = Image.open(img_location)  # open image
    image = image.rotate(270, expand=True)  # rotate image
    return image

def resize_image(image, base_height=600):
    hpercent = (base_height/float(image.size[1]))
    wsize = int((float(image.size[0])*float(hpercent)))
    image = image.resize((wsize,base_height), Image.ANTIALIAS)
    return image

def resize_image_width(image, base_width=300):
    wpercent = (base_width/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((base_width,hsize), Image.ANTIALIAS)
    return image
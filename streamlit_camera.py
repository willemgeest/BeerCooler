import streamlit_camera as st
import pandas as pd
import numpy as np
import urllib.request
from PIL import Image
import os
import torch
from torchvision import models, transforms
from torch import nn
import get_image
import object_detection
#import image_classification
import beer_classification

st.set_page_config(layout="wide")
st.header('Beer bottle classification algorithm')

model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_picture/latest_camera_photo_scored.jpg'
class_names = object_detection.get_classes()
img_location = 'latest_picture/latest_camera_photo.jpg'
image = get_image.get_image(IPv4_adress='http://192.168.178.108:8080', img_location= img_location)
#image = get_image.get_image(IPv4_adress='http://192.168.2.7:8080', img_location= img_location)

st.text('Picture captured')

#object detection
obj_det_model = object_detection.get_obj_det_model()
try:
    image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=True)
except:  # if GPU fails, try CPU
    image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)

if n_beers > 0:
    image_scored.save(scored_image_location)
    description_objdet = 'Beer bottle detected'
else:
    description_objdet = 'No beer bottle detected'

#heatmap
if n_beers > 0:
    img_heatmap, probabilities, label = beer_classification.beer_classification(img_location='.\\latest_picture\\latest_camera_photo_scored.jpg',
                                                                    heatmap_location='.\\latest_picture\\heatmap.jpg')


column1, column2, column3, column4 = st.beta_columns(4)

with column1:
    st.image(image = get_image.resize_image(image), caption= 'Original picture')

with column2:
    st.image(get_image.resize_image(image_scored), caption= description_objdet)

with column3:
    if n_beers>0:
        probabilities = probabilities.tolist()[0]

        df = pd.DataFrame([round(num*100, 1) for num in probabilities], class_names)
        df.columns = ['(%)']
        logo_location = 'logos/' + str(label) + '.png'
        st.image(get_image.resize_image_width(Image.open(logo_location).convert('RGB')))

        #if label == 'hertogjan':
        #    st.image(get_image.resize_image_width(Image.open('logos/hertogjan.jpg')))
        #if label == 'amstel':
        #    st.image(get_image.resize_image_width(Image.open('logos/amstel.png')))
        #if label == 'heineken':
        #    st.image(get_image.resize_image_width(Image.open('logos/heineken.png')))

        st.text('Probabilities:')
        st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
    else:
        st.text('No beers detected')

with column4:
    if n_beers > 0:
        st.image(get_image.resize_image(img_heatmap), caption='Heatmap')


import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import object_detection
import beer_classification
import sys

st.set_page_config(layout="wide")
st.header("Willem's beer bottle classification algorithm")

model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_picture/latest_camera_photo_scored.jpg'
class_names = beer_classification.get_classes()
img_location = 'latest_picture/latest_camera_photo.jpg'

image = get_image.get_image_IPcamera(IPv4_adress=sys.argv[1], img_location= img_location)

st.text('Picture captured')

# object detection
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

# heatmap
if n_beers > 0:
    img_heatmap, probabilities, label = beer_classification.beer_classification(img_location='.\\latest_picture\\latest_camera_photo_scored.jpg',
                                                                    heatmap_location='.\\latest_picture\\heatmap.jpg')

# define 4 columns
column1, column2, column3, column4 = st.beta_columns(4)

with column1:
    st.image(image = get_image.resize_image(image, max_width=400, max_heigth=600), caption= 'Original picture')

with column2:
    st.image(get_image.resize_image(image_scored, max_width=400, max_heigth=600), caption= description_objdet)

with column3:
    if n_beers>0:
        probabilities = probabilities.tolist()[0]

        df = pd.DataFrame([round(num*100, 1) for num in probabilities], class_names)
        df.columns = ['(%)']
        logo_location = 'logos/' + str(label) + '.png'
        st.image(get_image.resize_image(Image.open(logo_location).convert('RGB'), max_width=400, max_heigth=600))

        st.text('Probabilities:')
        st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
    else:
        st.text('No beers detected')

with column4:
    if n_beers > 0:
        st.image(get_image.resize_image(img_heatmap, max_width=400, max_heigth=600), caption='Heatmap')


import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import object_detection
import beer_classification
import os
import glob
import shutil

st.set_page_config(layout="wide", page_title="Beer bottle analyzing application")
st.header('Beer bottle analyzing application')
st.markdown("Check [Github](https://github.com/willemgeest/BeerCooler/tree/webapp) for more info")

device = st.radio("Select your device", ('Phone', 'Laptop / pc'))

# set parameters
model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_uploaded_photo_scored.jpg'
class_names = beer_classification.get_classes()
img_location = 'latest_camera_photo.jpg'

# upload file
image = st.file_uploader("Please upload your beer picture here", type= ['jpeg', 'jpg'])

if image is not None:
    image = Image.open(image)
    if device=='Phone' and image.size[0] > image.size[1]: # phone and landscape
        image = image.rotate(270)
    #st.markdown('**Original picture**')
    #st.image(image=get_image.resize_image(image=image, max_width=1200, max_heigth=600))

    with st.spinner('Image is being analyzed... Please wait a few seconds...'):

        if device=='Laptop / pc':
            # define 4 columns
            column1, column2, column3, column4 = st.beta_columns(4)

            with column1:
                # plot original image
                st.markdown('**Original image:**')
                st.image(get_image.resize_image(image=image, max_width=400, max_heigth=600))
        else:
            st.markdown('**Original image:**')
            st.image(get_image.resize_image(image=image, max_width=400, max_heigth=600))

        #object detection
        #obj_det_model = object_detection.get_obj_det_model(local=True)
        obj_det_model = object_detection.get_obj_det_model_Drive()
        #try:
        #    image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=True)
        #except:  # if GPU fails, try CPU
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)

        if n_beers > 0:
            image_scored.save(scored_image_location)

            img_heatmap, probabilities, label = beer_classification.beer_classification(img_location=scored_image_location,
                                                                                        heatmap_location='heatmap_uploaded.jpg')
        if device=='Laptop / pc':
            with column2:
                if n_beers > 0:
                    st.markdown('**Detected beer bottle:**')
                    st.image(get_image.resize_image(image=image_scored, max_width=400, max_heigth=400 ))
                else:
                    st.markdown('**Detecting beer bottles (no beers detected)**')
        else:
            if n_beers > 0:
                st.markdown('**Detected beer bottle:**')
                st.image(get_image.resize_image(image=image_scored, max_width=400, max_heigth=400))
            else:
                st.markdown('**No beers detected)**')

        if n_beers > 0:
            # logo
            logo_location = 'logos/' + str(label.lower()) + '.png'
            # create df with probabilities
            probabilities = probabilities.tolist()[0]
            df = pd.DataFrame([round(num * 100, 1) for num in probabilities], class_names)
            df.columns = ['(%)']

        if device == 'Laptop / pc':
            with column3:
                if n_beers>0:
                    st.markdown('**Predicted beer brand:**')
                    st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=300, max_heigth=300))
                    st.markdown('**Probabilities:**')
                    st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
                else:
                    st.markdown('**Predicting beer brands (no beers detected)**')
        else:
            if n_beers > 0:
                st.markdown('**Predicted beer brand:**')
                st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=300,
                                                max_heigth=300))
                st.markdown('**Probabilities:**')
                st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

        if device == 'Laptop / pc':
            with column4:
                if n_beers > 0:
                    # heatmap
                    st.markdown("**Heatmap (what makes algorithm think it's " + str(label) + '?)**')
                    st.image(get_image.resize_image(image=img_heatmap, max_width=400, max_heigth=400))
                else:
                    st.markdown("**Heatmap (no beers detected)**")
        else:
            if n_beers > 0:
                # heatmap
                st.markdown("**Heatmap (what makes algorithm think it's " + str(label) + '?)**')
                st.image(get_image.resize_image(image=img_heatmap, max_width=400, max_heigth=400))

        shutil.rmtree('checkpoints')

import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import object_detection
import beer_classification
import GD_download

st.set_page_config(layout="wide")
st.header("Willem's beer bottle classification algorithm")

# set parameters
model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_picture/latest_uploaded_photo_scored.jpg'
class_names = beer_classification.get_classes()
img_location = 'latest_picture/latest_camera_photo.jpg'

#download beer classification model from Google Drive (if not already available)
GD_download.get_beerclass_model_Drive(modelname=model_name)

# upload file
image = st.file_uploader("Please upload your beer picture here")

if image is not None:
    image = Image.open(image)
    st.markdown('**Original picture**')
    st.image(image=get_image.resize_image(image=image, max_width=1200, max_heigth=600))

    #object detection
    obj_det_model = object_detection.get_obj_det_model()
    try:
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=True)
    except:  # if GPU fails, try CPU
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)

    if n_beers > 0:
        image_scored.save(scored_image_location)

        img_heatmap, probabilities, label = beer_classification.beer_classification(img_location='.\\latest_picture\\latest_uploaded_photo_scored.jpg',
                                                                                    heatmap_location='.\\latest_picture\\heatmap_uploaded.jpg')
    # define 3 columns
    column1, column2, column3 = st.beta_columns(3)

    with column1:
        if n_beers > 0:
            st.markdown('**Detected beer bottle:**')
            st.image(get_image.resize_image(image=image_scored, max_width=500, max_heigth=600 ))
        else:
            st.markdown('**Detecting beer bottles (no beers detected)**')
    with column2:
        if n_beers>0:
            st.markdown('**Predicted beer brand:**')
            # logo
            logo_location = 'logos/' + str(label) + '.png'
            st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=500, max_heigth=600))

            # create df with probabilities
            probabilities = probabilities.tolist()[0]
            df = pd.DataFrame([round(num*100, 1) for num in probabilities], class_names)
            df.columns = ['(%)']
            st.markdown('**Probabilities:**')
            st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
        else:
            st.markdown('**Predicting beer brands (no beers detected)**')

    with column3:
        if n_beers > 0:
            # heatmap
            st.markdown("**Heatmap (what makes algorithm think it's " + str(label) + '?)**')
            st.image(get_image.resize_image(image=img_heatmap, max_width=500, max_heigth=600))
        else:
            st.markdown("**Heatmap (no beers detected)**")


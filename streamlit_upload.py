import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import object_detection
import gradCAM

st.set_page_config(layout="wide")
st.header('Beer bottle classification algorithm')

model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_picture/latest_uploaded_photo_scored.jpg'
class_names = object_detection.get_classes()
img_location = 'latest_picture/latest_camera_photo.jpg'

image = st.file_uploader("Please upload your beer picture here")

if image is not None:
    image = Image.open(image)
    #image = get_image.get_image(IPv4_adress='http://192.168.178.108:8080', img_location= img_location)
    #image = get_image.get_image(IPv4_adress='http://192.168.2.7:8080', img_location= img_location)

    #st.text('Picture captured')

    #object detection
    obj_det_model = object_detection.get_obj_det_model()
    try:
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=True)
    except:  # if GPU fails, try CPU
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)

    if n_beers > 0:
        image_scored.save(scored_image_location)

    #heatmap
    if n_beers > 0:
        img_heatmap, probabilities, label = gradCAM.heatmap(img_location='.\\latest_picture\\latest_uploaded_photo_scored.jpg',
                                                            heatmap_location='.\\latest_picture\\heatmap_uploaded.jpg')
    st.markdown('**Original picture**')
    st.image(image=get_image.resize_image(image))

    column1, column2, column3 = st.beta_columns(3)

    with column1:
        if n_beers > 0:
            st.markdown('**Detected beer bottle:**')
            st.image(get_image.resize_image(image_scored))
        else:
            st.markdown('**Detecting beer bottles (no beers detected)**')
    with column2:
        if n_beers>0:
            st.markdown('**Predicted beer brand:**')
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

            st.markdown('**Probabilities:**')
            st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
        else:
            st.markdown('**Predicting beer brands (no beers detected)**')

    with column3:
        if n_beers > 0:
            st.markdown("**Heatmap (what makes algorithm think it's " + str(label) + '?)**')
            st.image(get_image.resize_image(img_heatmap))
        else:
            st.markdown("**Heatmap (no beers detected)**")


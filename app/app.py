import os

import streamlit as st

from app.predictor import get_class_name, predict
from app.utils import get_asset_path, save_uploaded_file

st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

logo_path = get_asset_path('Disease.png')
if logo_path:
    st.image(str(logo_path))
else:
    st.warning('Logo image not found in assets/Disease.png.')

if app_mode == 'Home':
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Prediction System for Sustainable Agriculture')
    uploaded_file = st.file_uploader('Choose an Image:')

    save_path = None
    if uploaded_file is not None:
        save_path = save_uploaded_file(uploaded_file, os.path.join(os.getcwd(), 'test_image.jpg'))
        st.image(uploaded_file, width=400, use_container_width=True)

    if st.button('Predict'):
        if save_path is None:
            st.warning('Please upload an image before predicting.')
        else:
            st.write('Our Prediction:')
            try:
                result_index = predict(str(save_path))
                st.success(f'Model is predicting that it is {get_class_name(result_index)}')
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')

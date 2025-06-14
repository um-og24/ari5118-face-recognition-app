import sys
sys.dont_write_bytecode = True

import streamlit as st

from services.api_services import upload_from_local
from services.io_services import IMAGE_FILE_TYPES


def show_component(caller_name: str, api_url: str, func, data = None, ground_truth: str = None):
    uploaded_file = st.file_uploader("Choose an image", type=IMAGE_FILE_TYPES, key=f"{caller_name}_upload_image_file")
    if uploaded_file is None:
        return

    try:
        with st.spinner("uploading image, please wait..."):
            image_bytes, image_file = upload_from_local(uploaded_file)

        if ground_truth is None:
            func(api_url, image_bytes, image_file, data)
        else:
            func(api_url, image_bytes, image_file, data, ground_truth)
    except Exception as e:
        st.error(f"ERROR: {e}")

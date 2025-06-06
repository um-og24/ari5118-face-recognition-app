import sys
sys.dont_write_bytecode = True

import streamlit as st

from services.api_services import fetch_image_from_url


def show_component(caller_name: str, api_url: str, func, data = None, ground_truth: str = None):
    image_url = st.text_input("Please provide a direct link to the image:", key=f"{caller_name}_upload_web_image_link")
    #value="https://i.pinimg.com/736x/73/e0/29/73e0291f5cc98c89e16bb544857b4e77.jpg"
    #value="https://cdn.shopify.com/s/files/1/1140/8354/files/Black_Widow_Marvel_Timeline.jpg?v=1631858232"
    #value="https://scontent.fmla2-1.fna.fbcdn.net/v/t39.30808-6/482222794_615961994685433_5089288090154051648_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=833d8c&_nc_ohc=GwVTc3U6YFcQ7kNvwEdaXRQ&_nc_oc=Adk-uSwjTUjW9JcEym6sJ1bqugwQhY5Ei56J9gByiZSHhmjoquRhRgqzDlvz5czLW0w&_nc_zt=23&_nc_ht=scontent.fmla2-1.fna&_nc_gid=1rDu_oMj0K2b7YPVsb44QA&oh=00_AfEuszxg2HP3HKhwqZbFf5wrTZ5WIwUE3EaMcBKtKmV9nQ&oe=6806197D"
    if len(image_url) == 0:
        return
    elif not image_url.startswith(("http://", "https://")):
        st.error("The link provided is not valid.")
        return

    try:
        with st.spinner("downloading and processing image, please wait..."):
            image_bytes, image_file = fetch_image_from_url(image_url)

        if ground_truth is None:
            func(api_url, image_bytes, image_file, data)
        else:
            func(api_url, image_bytes, image_file, data, ground_truth)
    except Exception as e:
        st.error(f"ERROR: {e}")

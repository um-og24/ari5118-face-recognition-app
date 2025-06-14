import sys
sys.dont_write_bytecode = True

import streamlit as st
import cv2
import numpy as np
import tempfile

from utilities import utils
from services import http_services, api_services
from utilities.configs import FRAME_SKIP, DETECT_API_URL, UPLOAD_CAPTURE_API_URL
from helpers.video_processor import VideoProcessor
from ui_components import image_file_uploader_ui, image_web_link_ui
from ui_components.gallery_ui import show_gallery, plot_detection_results, center_an_image

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from fastapi.datastructures import UploadFile


def main_contribution(operation_mode: str, library: str, detector_name: str, model_name: str, name: str = ""):
    if operation_mode == "detection":
        www_image_link_tab, upload_image_tab, webcam_tab, m3u8_stream_tab = st.tabs(["🌐 Web Image Link", "📤 Upload Image", "📷 Webcam (Live Feed)", "📺 Stream (M3U8)"])

        with m3u8_stream_tab:
            _main_m3u8_streaming_function(library, detector_name, model_name)
    else:
        www_image_link_tab, upload_image_tab, webcam_tab, gallery_tab = st.tabs(["🌐 Web Image Link", "📤 Upload Image", "📷 Webcam (Live Feed)", f"🗄️ Manage Contributions"])

        with gallery_tab:
            show_gallery(name, name)

    with www_image_link_tab:
        _main_web_image_link_function(operation_mode, library, detector_name, model_name, name)

    with upload_image_tab:
        _main_upload_function(operation_mode, library, detector_name, model_name, name)

    with webcam_tab:
        _main_webcam_function(operation_mode, library, detector_name, model_name, name)


def _main_upload_function(operation_mode: str, library: str, detector_name: str, model_name: str, name: str = ""):
    if "detection" in operation_mode:
        api_url = DETECT_API_URL
        data={"library": library, "detector_name": detector_name, "model_name": model_name}
    elif "contribution" in operation_mode:
        api_url = UPLOAD_CAPTURE_API_URL
        data={"library": library, "detector_name": detector_name, "model_name": model_name, "name": name}
    else:
        st.error("Unsupported mode of execution.")
        return

    image_file_uploader_ui.show_component(operation_mode, api_url, _perform_and_plot_results, data)

def _main_web_image_link_function(operation_mode, library: str, detector_name: str, model_name: str = "", name: str = ""):
    if "detection" in operation_mode:
        api_url = DETECT_API_URL
        data={"library": library, "detector_name": detector_name, "model_name": model_name}
    elif "contribution" in operation_mode:
        api_url = UPLOAD_CAPTURE_API_URL
        data={"library": library, "detector_name": detector_name, "model_name": model_name, "name": name}
    else:
        st.error("Unsupported mode of execution.")
        return

    image_web_link_ui.show_component(operation_mode, api_url, _perform_and_plot_results, data)


def _main_webcam_function(operation_mode: str, library: str, detector_name: str, model_name: str, name: str = ""):
    if "detection" in operation_mode:
        key="demo_live_capture"
        data={"library": library, "detector_name": detector_name, "model_name": model_name}
        vid_processor_facotry = lambda: VideoProcessor(api_url=DETECT_API_URL, data=data, model_name=model_name)
    elif "contribution" in operation_mode:
        key="contribution_live_capture"
        data={"library": library, "detector_name": detector_name, "model_name": model_name, "name": name}
        vid_processor_facotry = lambda: VideoProcessor(api_url=UPLOAD_CAPTURE_API_URL, data=data, model_name=model_name, limit_frames=True)
    else:
        st.error("Unsupported mode of execution.")
        return

    webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=vid_processor_facotry,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

def _main_m3u8_streaming_function(library: str, detector_name: str, model_name: str):
    st.subheader("📺 Stream from YouTube, .m3u8 link or File, and more")
    stream_url = st.text_input("Enter .m3u8 URL")
    uploaded_m3u8 = st.file_uploader("Or upload an .m3u8 file", type="m3u8")

    left_col, _, right_col = st.columns([1,2,1])

    #if st.session_state.get("streaming", False):
    with right_col:
        if st.button("Stop Stream", type="primary", use_container_width=True):
            st.session_state.streaming = False
    #else:
    with left_col:
        if st.button("Start Stream", type="primary", use_container_width=True):
            st.session_state.streaming = True

    if st.session_state.get("streaming", False):
        if uploaded_m3u8:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m3u8") as tmp_file:
                tmp_file.write(uploaded_m3u8.read())
                stream_url = tmp_file.name

        if not stream_url:
            st.error("Please enter a stream URL or upload a file.")
            return

        if "youtube.com" in stream_url or "youtu.be" in stream_url:
            stream_url = utils.get_direct_youtube_url(stream_url)

        frame_count = 0
        cap = cv2.VideoCapture(stream_url)
        st_frame = st.empty()

        # Target resolution for processing
        #target_width, target_height = 320, 240
        target_width, target_height = 640, 480
        #target_width, target_height = 1024, 768

        while cap.isOpened() and st.session_state.get("streaming", False):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for processing while maintaining aspect ratio
            original_frame, processed_frame = utils.resize_frame_for_api(frame, target_width, target_height)

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                _, img_encoded = cv2.imencode(".jpg", processed_frame)
                response = http_services.post_request(
                    DETECT_API_URL,
                    files={"file": (f"frame_{frame_count}.jpg", img_encoded.tobytes(), "image/jpeg")},
                    data={"library": library, "detector_name": detector_name, "model_name": model_name}
                )
                if response.ok:
                    boxes = response.json().get("boxes", [])
                    names = response.json().get("names", [])
                    
                    # # Scale bounding boxes back to original frame size if needed
                    # scaled_boxes = utils.scale_boxes_to_original(
                    #     boxes, 
                    #     processed_frame.shape[:2], 
                    #     original_frame.shape[:2]
                    # )
                    original_frame = processed_frame
                    scaled_boxes = boxes

                    utils.plot_bounding_boxes(original_frame, scaled_boxes, names)
                    with st_frame:
                        center_an_image(original_frame[:, :, ::-1], channels="RGB", caption=f"Captured Frame #{frame_count} / Feed @ {FRAME_SKIP}FPS")

        cap.release()


def _perform_and_plot_results(api_url: str, image_bytes: bytes, image_file: UploadFile, data = None) -> None:
    with st.spinner("recognition in process, please wait..."):
        response_data = api_services.perform_face_recognition_task(api_url, image_file, data)

    with st.spinner("displaying results, please wait..."):
        #st.write(response_data)
        if "error" in response_data:
            st.error(f"ERROR: {response_data['error']}")
        else:
            _plot_results(image_bytes, response_data)

def _plot_results(image_bytes: bytes, response_data):
    img = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #_, img_encoded = cv2.imencode(".jpg", img)

    original_image = img.copy()

    boxes = response_data["boxes"]
    names = response_data["names"]

    with st.expander("**Original Image**", expanded=True, icon="🖼️"):
        utils.plot_bounding_boxes(img, boxes, names)
        center_an_image(img[:, :, ::-1], channels="RGB", caption="Detection Result")

    plot_detection_results(original_image, response_data)

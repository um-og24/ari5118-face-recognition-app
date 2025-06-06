import sys
sys.dont_write_bytecode = True

import cv2
import numpy as np
import threading
import time
import services.http_services as http_services
from typing import Callable
from streamlit_webrtc import VideoProcessorBase
from services.http_services import post_request
from utilities.utils import plot_bounding_boxes
from av import VideoFrame


# Live Stream Handling
class StreamingProcessor(VideoProcessorBase):
    def __init__(self, api_url: str, data, model_name: str, mirror_image: bool = False, stream_url: str = None):
        self.api_url = api_url
        self.data = data
        self.model_name = model_name
        self.mirror_image = mirror_image
        self.frame_count = 0
        self.stream_url = stream_url
        self.cap = None

    def recv(self, frame):
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            ret, img = self.cap.read()
            if not ret:
                return frame

            if self.mirror_image:
                img = cv2.flip(img, 1) # mirror image
            self.frame_count += 1

            # Encode and send to API
            _, img_encoded = cv2.imencode(".jpg", img)
            response = post_request(
                self.api_url,
                files={"file": (f"frame_{self.frame_count}.jpg", img_encoded.tobytes(), "image/jpeg")},
                data=self.data
            )

            # If API responds OK, draw boxes
            if response.ok:
                boxes = response.json().get("boxes", [])
                names = response.json().get("names", [])
                plot_bounding_boxes(img, boxes, names)

            return VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame


    def update(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                else:
                    # Handle stream interruption
                    self.cap.release()
                    time.sleep(1)  # Wait before trying to reconnect
                    self.cap = cv2.VideoCapture(self.stream_url)
            time.sleep(0.01)  # Short sleep to prevent CPU overload
    
    def transform(self, frame):
        if self.last_frame is not None:
            # Process the frame for face recognition (resize to 1024x768)
            processed_frame = self.processor.process_frame_for_recognition(self.last_frame)
            
            # Send to face recognition API
            # recognition_result = self.send_to_api(processed_frame)
            
            # Draw results on the original frame for display
            # annotated_frame = self.draw_recognition_results(self.last_frame, recognition_result)
            
            # For demonstration, just return the processed frame
            return processed_frame
        return frame
    
    def send_to_api(self, frame):
        """Send the processed frame to the face recognition API"""
        # Convert the frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        
        # Create headers and data for the API request
        headers = {'Content-Type': 'image/jpeg'}
        data = jpeg.tobytes()
        
        try:
            # Replace with your actual API endpoint
            response = http_services.post_request('https://your-face-recognition-api.com/recognize', 
                                    headers=headers, 
                                    data=data, 
                                    timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"API Connection Error: {str(e)}")
            return None
    
    def draw_recognition_results(self, original_frame, results):
        """Draw recognition results on the original frame"""
        if results is None:
            return original_frame
            
        # Clone the frame to avoid modifying the original
        annotated_frame = original_frame.copy()
        
        # Draw the results on the frame
        # This depends on your API response format
        # Example:
        # for face in results['faces']:
        #     x, y, w, h = face['bbox']
        #     name = face['name']
        #     cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     cv2.putText(annotated_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return annotated_frame
    
    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()



class M3U8StreamProcessor:
    def __init__(self, stream_url: str, model_name: str, target_size=(640, 480)):
        self.stream_url = stream_url
        self.model_name = model_name
        self.target_width, self.target_height = target_size
        self.running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0

    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        target_aspect = self.target_width / self.target_height
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = self.target_width
            new_h = int(new_w / current_aspect)
        else:
            new_h = self.target_height
            new_w = int(new_h * current_aspect)

        resized_frame = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        return canvas

    def _stream_loop(self, display_callback: Callable[[np.ndarray], None]):
        self.cap = cv2.VideoCapture(self.stream_url)
        self.frame_count = 0

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            resized_frame = self.resize_frame(frame)
            self.frame_count += 1

            _, img_encoded = cv2.imencode(".jpg", resized_frame)
            response = post_request(
                "https://facerecapi.oghomelabs.com/detect",
                files={"file": (f"frame_{self.frame_count}.jpg", img_encoded.tobytes(), "image/jpeg")},
                data={"model": self.model_name}
            )

            if response.ok:
                boxes = response.json().get("boxes", [])
                names = response.json().get("names", [])
                plot_bounding_boxes(resized_frame, boxes, names)

            display_callback(resized_frame[:, :, ::-1])  # Convert BGR to RGB for display

        if self.cap:
            self.cap.release()

    def start(self, display_callback: Callable[[np.ndarray], None]):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, args=(display_callback,), daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None




# def main_streaming_function(operation_mode: str, model_name: str, name: str = ""):
#     st.subheader("📺 Stream from .m3u8 Link or File")
#     stream_url = st.text_input("Enter .m3u8 URL")
#     uploaded_m3u8 = st.file_uploader("Or upload an .m3u8 file", type="m3u8")

#     if uploaded_m3u8:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".m3u8") as tmp_file:
#             tmp_file.write(uploaded_m3u8.read())
#             stream_url = tmp_file.name

#     if not stream_url:
#         st.error("Please enter a stream URL or upload a file.")
#         return

#     if operation_mode == "detection":
#         key="demo_streaming_capture"
#         data={"model": model_name}
#         vid_processor_facotry = lambda: StreamingProcessor(api_url=DETECT_API_URL, data=data, model_name=model_name, stream_url=stream_url)
#     else:
#         key="contribution_streaming_capture"
#         data={"model": model_name, "name": name}
#         vid_processor_facotry = lambda: StreamingProcessor(api_url=UPLOAD_CAPTURE_API_URL, data=data, model_name=model_name, stream_url=stream_url)

#     webrtc_streamer(
#         key=key,
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=vid_processor_facotry,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True
#     )

# def m3u8_stream_view(model_name: str):
#     st.subheader("📺 Stream from .m3u8 Link or File")

#     # Inputs for user stream
#     stream_url = st.text_input("Enter .m3u8 URL")
#     uploaded_m3u8 = st.file_uploader("Or upload an .m3u8 file", type="m3u8")

#     # Streaming container
#     st_frame = st.empty()

#     # Use session state to persist processor
#     if "m3u8_processor" not in st.session_state:
#         st.session_state.m3u8_processor = None

#     # Convert file upload to temporary file
#     if uploaded_m3u8:
#         import tempfile
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".m3u8") as tmp_file:
#             tmp_file.write(uploaded_m3u8.read())
#             stream_url = tmp_file.name

#     # Start/Stop Buttons
#     left_col, _, right_col = st.columns([2, 6, 2])
#     with left_col:
#         if st.button("Start Stream", type="primary", use_container_width=True):
#             if stream_url:
#                 def display_callback(frame_rgb):
#                     st_frame.image(frame_rgb, channels="RGB")

#                 st.session_state.m3u8_processor = M3U8StreamProcessor(
#                     stream_url=stream_url,
#                     model_name=model_name,
#                     target_size=(640, 480)
#                 )
#                 st.session_state.m3u8_processor.start(display_callback)
#                 st.session_state.streaming = True
#             else:
#                 st.error("Please enter a valid stream URL or upload a file.")

#     with right_col:
#         if st.button("Stop Stream", type="primary", use_container_width=True):
#             if st.session_state.m3u8_processor:
#                 st.session_state.m3u8_processor.stop()
#                 st.session_state.m3u8_processor = None
#                 st.session_state.streaming = False




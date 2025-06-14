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

# Webcam Stream Handling
class VideoProcessor(VideoProcessorBase):
    def __init__(self, api_url: str, data, model_name: str, limit_frames: bool = False, mirror_image: bool = False):
        self.api_url = api_url
        self.data = data
        self.model_name = model_name
        self.limit_frames = limit_frames
        self.mirror_image = mirror_image
        self.frame_count = 0
        self.max_frames = 200
        
        # Target resolution for processing
        self.target_width = 640
        self.target_height = 480

    def resize_frame(self, frame):
        """Resize frame to target resolution while maintaining aspect ratio"""
        # Get current frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate the scaling factor to fit within target dimensions
        # while preserving aspect ratio
        target_aspect = self.target_width / self.target_height
        current_aspect = w / h
        
        if current_aspect > target_aspect:
            # Width is the limiting factor
            new_w = self.target_width
            new_h = int(new_w / current_aspect)
        else:
            # Height is the limiting factor
            new_h = self.target_height
            new_w = int(new_h * current_aspect)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Create a black canvas of the target size
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image on the canvas
        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        return canvas

    def recv(self, frame):
        try:
            if self.limit_frames and self.frame_count >= self.max_frames:
                return frame  # No more processing

            img = frame.to_ndarray(format="bgr24")
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



class M3U8Streamer(VideoProcessorBase):
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.processor = VideoProcessor()
        self.cap = None
        self.stopped = False
        self.last_frame = None
        
        # Start capturing in a separate thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
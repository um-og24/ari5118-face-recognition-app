import sys
sys.dont_write_bytecode = True

import cv2
import json
import numpy as np

from fastapi.datastructures import UploadFile
from io import BytesIO

from services.http_services import get_request, post_request
from utilities.configs import TRAIN_API_URL, DETECT_AND_COMPARE_API_URL

def fetch_image_from_url(url: str) -> (bytes, UploadFile):
    response = get_request(url, stream=True)
    image_bytes = response.content
    filename = url.split('/')[-1].split('?')[0] or "downloaded_image.jpg"
    image_object = BytesIO(image_bytes)
    return image_bytes, UploadFile(file=image_object, filename=filename)

def upload_from_local(uploaded_file) -> (bytes, UploadFile):
    img = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    _, img_encoded = cv2.imencode(".jpg", img)
    image_object = BytesIO(img_encoded)
    return img_encoded, UploadFile(file=image_object, filename=uploaded_file.name)

def perform_face_recognition_task(api_url: str, file: UploadFile, data=None):
    if data is None:
        data = {}
    files = {"file": (file.filename, file.file, "image/jpeg")}
    response = post_request(api_url, files=files, data={k: v for k, v in data.items() if v is not None})
    return json.loads(response.content) if response.ok else json.loads(response.content)

def perform_retraining_of_models(deepface_pairs: list = None, facerecognition_models: list = None, insightface_models: list = None):
    data = {
        "deepface_detector_model_pairs": ",".join(deepface_pairs) if deepface_pairs else None,
        "facerecognition_models": ",".join(facerecognition_models) if facerecognition_models else None,
        "insightface_models": ",".join(insightface_models) if insightface_models else None
    }
    response = post_request(TRAIN_API_URL, data={k: v for k, v in data.items() if v is not None})
    return json.loads(response.content) if response.ok else json.loads(response.content)
import sys
sys.dont_write_bytecode = True

from os import makedirs

GALLERY_MASTER_PASSWORD = "LetMeSeeThemPlz2025!."

DATA_DIR = "data"
makedirs(DATA_DIR, exist_ok=True)

DATASET_DIR = f"{DATA_DIR}/dataset"
makedirs(DATASET_DIR, exist_ok=True)

LFW_DATASET_DIR = f"{DATA_DIR}/lfw_dataset"
makedirs(LFW_DATASET_DIR, exist_ok=True)

TEST_DATASET_DIR = f"{DATA_DIR}/test_dataset"
makedirs(DATASET_DIR, exist_ok=True)

TEST_DATASET_JSON_PATH = f"{TEST_DATASET_DIR}/test_dataset.json"

BENCHMARKS_REPORTS_DIR = f"{DATA_DIR}/benchmarks"
makedirs(BENCHMARKS_REPORTS_DIR, exist_ok=True)

FRAME_SKIP = 5  # Skip every N-th frame for analysis

SYSTEM_MONITORING_URL = "http://localhost:8502"
#SYSTEM_MONITORING_URL = "https://sysmon.oghomelabs.com"

_BASE_API_URL = "http://localhost:8000"
#_BASE_API_URL = "https://facerec-api.oghomelabs.com"
DETECT_API_URL = f"{_BASE_API_URL}/detect"
DETECT_AND_COMPARE_API_URL = f"{_BASE_API_URL}/detect_and_compare"
TRAIN_API_URL = f"{_BASE_API_URL}/train_model"
UPLOAD_CAPTURE_API_URL = f"{_BASE_API_URL}/capture"
REMOVE_CONTRIBUTION_API_URL = f"{_BASE_API_URL}/remove_asset"


SUPPORTED_LIBRARIES = [
    "DeepFace",
    "FaceRecognition",
    "InsightFace"
]


FACE_RECOGNITION_SUPPORTED_MODELS = [
    "hog",
    # "cnn",
]


DEEPFACE_SUPPORTED_BACKENDS_DETECTORS = [
    "retinaface",
    "opencv",
    "ssd",
    #"dlib",
    "mtcnn",
    #"fastmtcnn",
    #"mediapipe",
    #"yolov8",
    #"yolov11s",
    #"yolov11n",
    #"yolov11m",
    "yunet",
    #"centerface",
]

DEEPFACE_SUPPORTED_MODELS = [
    "VGG-Face",
    "ArcFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "Facenet",
    "Facenet512",
    "GhostFaceNet",
    #"Buffalo_L",
    "OpenFace",
    "SFace",
]


INSIGHTFACE_SUPPORTED_MODELS = [
    "buffalo_l"
]
import sys
sys.dont_write_bytecode = True

from os import makedirs

DATA_DIR = "data"
makedirs(DATA_DIR, exist_ok=True)

DATASET_DIR = f"{DATA_DIR}/dataset"
makedirs(DATASET_DIR, exist_ok=True)

LFW_DATASET_DIR = f"{DATA_DIR}/lfw_dataset"
makedirs(LFW_DATASET_DIR, exist_ok=True)

UPLOADS_DIR = f"{DATA_DIR}/uploads"
makedirs(UPLOADS_DIR, exist_ok=True)

MODELS_DIR = f"{DATA_DIR}/models"
makedirs(MODELS_DIR, exist_ok=True)

EMBEDDINGS_DIR = f"{DATA_DIR}/embeddings"
makedirs(EMBEDDINGS_DIR, exist_ok=True)


SUPPORTED_LIBRARIES = [
    "DeepFace",
    "FaceRecognition",
    "InsightFace"
]



FACE_RECOGNITION_SUPPORTED_MODELS = [
    "hog",
    #"cnn",
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

DEEPFACE_SUPPORTED_METRICS = [
    "cosine",
    "euclidean",
    "euclidean_l2",
    "angular"
]

DEEPFACE_SUPPORTED_FACIAL_ANALYSIS = [
    "age",
    "gender",
    "race",
    "emotion"
]


INSIGHTFACE_SUPPORTED_MODELS = [
    "buffalo_l"
]
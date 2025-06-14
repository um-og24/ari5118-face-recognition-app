import sys
sys.dont_write_bytecode = True

import cv2
import face_recognition

from utilities.configs import DATASET_DIR, EMBEDDINGS_DIR


def recognize_faces_with_face_recognition(face_image, model_name: str) -> (list[int], list[str],float):
    from time import time

    start = time()
    boxes, names = _recognize_faces(face_image, model_name)

    return boxes, names, time() - start


def _recognize_faces(rgb_image, model_name: str) -> (list[int], list[str]):
    names = []
    boxes = face_recognition.face_locations(rgb_image, model=model_name)
    encodings = face_recognition.face_encodings(rgb_image, boxes)
    data = _load_face_recognition_encodings(model_name)
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "unknown"

        if True in matches:
            matched_idxs = [i for i, match in enumerate(matches) if match]
            name_counts = {data["names"][i]: matched_idxs.count(i) for i in matched_idxs}
            name = max(name_counts, key=name_counts.get)

        names.append(name)

    return boxes, names


def _load_face_recognition_encodings(model_name: str):
    # Load the face recognition model
    try:
        from os import makedirs
        makedirs(f"{EMBEDDINGS_DIR}/FaceRecognition", exist_ok=True)
        encodings_filepath=f"{EMBEDDINGS_DIR}/FaceRecognition/{model_name}_encodings.pickle"

        from services.io_services import exists
        if not exists(encodings_filepath):
            train_face_recognition_model(DATASET_DIR, model_name)

        from pickle import loads
        model_encodings = loads(open(encodings_filepath, "rb").read())
    except Exception as e:
        # If no model is trained yet, just display the frame without recognition
        model_encodings = {"encodings": [], "names": []}
    return model_encodings


def train_face_recognition_model(dataset_dir: str, model_name: str) -> None:
    """
    Train a face recognition model using FaceRecognition (Dlib)

    dataset_dir: Directory containing subdirectories with person names and their images
    model_name: The actual FaceRecognition model to use (HOG or CNN)
    """
    try:
        # Print the model being used
        print(f"Training with FaceRecognition (Dlib) model: {model_name}")

        from imutils import paths
        imagePaths = list(paths.list_images(dataset_dir))
        knownEncodings = []
        knownNames = []

        from os import path
        for (i, imagePath) in enumerate(imagePaths):
            print(f"Processing image {i+1}/{len(imagePaths)}: {imagePath}")
            name = imagePath.split(path.sep)[-2]

            image = cv2.imread(imagePath)
            if image is None:
                print(f"Could not read image: {imagePath}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model=model_name)
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings is None or len(encodings) == 0:
                print(f"No faces found in: {imagePath}")
                continue

            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        from os import makedirs
        makedirs(f"{EMBEDDINGS_DIR}/FaceRecognition", exist_ok=True)

        from pickle import dumps
        data = {"encodings": knownEncodings, "names": knownNames}
        with open(f"{EMBEDDINGS_DIR}/FaceRecognition/{model_name}_encodings.pickle", "wb") as f:
            f.write(dumps(data))

        print(f"Training complete! {len(knownEncodings)} face encodings saved for {model_name}.")
    except Exception as e:
        print(f"Training error: {e}")
        return

import sys
sys.dont_write_bytecode = True

import os
import cv2
import json
import numpy as np

from insightface.app import FaceAnalysis

from utilities.configs import MODELS_DIR, EMBEDDINGS_DIR


# def recognize_faces_with_insightface(face_image, detector_backend: str, model_name: str):
#     """
#     Recognize faces using DeepFace's built-in functionality
    
#     face_image: Image containing faces to recognize
#     detector_backend: Face detector backend to use
#     model_name: The DeepFace model to use (ArcFace, Facenet, Facenet512, etc.)
#     """
#     from time import time
    
#     start = time()
#     boxes, names = recognize_faces(face_image, detector_backend, model_name)
    
#     return boxes, names, time() - start

def recognize_face(embedding, embedding_db, threshold=0.6):
    """
    Match embedding against database.

    Returns:
        (best_match_name, best_similarity) or (None, 0) if no match above threshold.
    """
    best_match = None
    best_score = 0

    for person, embeddings in embedding_db.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)
            if score > best_score and score > threshold:
                best_score = score
                best_match = person

    return best_match, best_score


def save_embedding_db(db, file_path):
    with open(file_path, 'w') as f:
        json.dump(db, f)

def load_embedding_db(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def train_insightface_model(dataset_path, recognition_model='buffalo_l'):
    """
    Build embeddings database from folder-structured dataset for InsightFace.

    Args:
        dataset_path (str): Root folder with subfolders per person.
        recognition_model (str): InsightFace recognition model to use.

    Returns:
        dict: {person_name: [embedding_vectors]}
    """
    app = FaceAnalysis(name=recognition_model, root=f"{MODELS_DIR}/insightface", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embedding_db = {}

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            # Take the first detected face embedding
            embedding = faces[0].embedding.tolist()
            embeddings.append(embedding)

        if embeddings:
            embedding_db[person_name] = embeddings

    from os import makedirs
    makedirs(f"{EMBEDDINGS_DIR}/Insightface", exist_ok=True)
    # Save the database as a JSON file
    db_path = f"{EMBEDDINGS_DIR}/Insightface/{recognition_model.lower()}_db.json"
    
    save_embedding_db(embedding_db, db_path)



import sys
sys.dont_write_bytecode = True

from utilities.configs import DATASET_DIR, MODELS_DIR, EMBEDDINGS_DIR

import os
# Set the custom directory where models will be downloaded
os.environ['DEEPFACE_HOME'] = f"{MODELS_DIR}/deepface"

import cv2
from deepface import DeepFace
import numpy as np
import json
from pathlib import Path


def recognize_faces_with_deepface(face_image, detector_backend: str, model_name: str):
    """
    Recognize faces using DeepFace's built-in functionality
    
    face_image: Image containing faces to recognize
    detector_backend: Face detector backend to use
    model_name: The DeepFace model to use (ArcFace, Facenet, Facenet512, etc.)
    """
    from time import time
    
    start = time()
    boxes, names = _recognize_faces(face_image, detector_backend, model_name)
    
    return boxes, names, time() - start


def _recognize_faces(rgb_image, detector_backend: str, model_name: str) -> (list[int], list[str]):
    """
    Recognize faces in an image using DeepFace
    
    Returns:
        boxes: List of face bounding boxes (top, right, bottom, left)
        names: List of recognized face names
    """
    names = []
    boxes = []

    try:
        # Ensure rgb_image is in the right format
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
        
        # First, detect faces in the image
        face_objs = DeepFace.extract_faces(
            img_path=rgb_image,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )
        
        # If no faces detected, return empty results
        if len(face_objs) == 0:
            return [], []
        
        # Prepare the path to the database
        db_path = f"{EMBEDDINGS_DIR}/DeepFace/{detector_backend.lower()}/{model_name.lower()}_db.json"
        
        # Check if database exists, if not, create it
        if not os.path.exists(db_path):
            train_deepface_model(DATASET_DIR, detector_backend, model_name)
        
        # If database file still doesn't exist, return empty results
        if not os.path.exists(db_path):
            return [], []
        
        # Load the database
        with open(db_path, 'r') as f:
            database = json.load(f)
        
        # Process each detected face
        for face_obj in face_objs:
            # Skip if confidence is too low
            if face_obj.get("confidence", 0) < 0.9:
                continue
            
            # Get face image and region
            face_img = face_obj["face"]
            facial_area = face_obj["facial_area"]
            
            # Convert to (top, right, bottom, left) format for return
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            box = (y, x + w, y + h, x)
            
            # Ensure face_img is in correct format for represent
            if isinstance(face_img, np.ndarray):
                if face_img.dtype != np.uint8:
                    face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
            
            # Generate embedding for this face
            embedding_result = DeepFace.represent(
                img_path=face_img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Extract the embedding vector
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                embedding = embedding_result[0].get("embedding", [])
            else:
                embedding = embedding_result.get("embedding", [])
            
            # Skip if embedding is empty
            if not embedding:
                continue
            
            # Compare with all known faces in the database
            best_match_name = "unknown"
            min_distance = 0.6  # Threshold for recognition
            
            # Find the best match for this face
            for person_name, person_data in database.items():
                for stored_embedding in person_data["embeddings"]:
                    # Calculate cosine similarity using DeepFace's distance calculation
                    from scipy.spatial.distance import cosine
                    distance = cosine(embedding, stored_embedding)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match_name = person_name
            
            # Add the results
            boxes.append(box)
            names.append(best_match_name)
    
    except Exception as e:
        print(f"Recognition error: {e}")
    
    return boxes, names


def train_deepface_model(dataset_dir: str, detector_backend: str, model_name: str) -> None:
    """
    Build a face database using DeepFace by storing face embeddings in a JSON file
    
    dataset_dir: Directory containing subdirectories with person names and their images
    detector_backend: Face detector backend to use
    model_name: The DeepFace model to use (ArcFace, Facenet, Facenet512, etc.)
    """
    try:
        # Print the model being used
        print(f"Building face database with DeepFace model: {model_name} using {detector_backend}")
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(f"{EMBEDDINGS_DIR}/DeepFace/{detector_backend.lower()}", exist_ok=True)
        
        # Initialize the database structure
        database = {}
        
        # Process each person's directory
        for person_dir in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_dir)
            if not os.path.isdir(person_path):
                continue
            
            # Initialize this person's data
            database[person_dir] = {
                "name": person_dir,
                "embeddings": [],
                "image_paths": []  # Store original image paths instead of duplicating images
            }
            
            # Process each image for this person
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(person_path, img_file)
                try:
                    # Extract face
                    face_objs = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    if len(face_objs) == 0:
                        print(f"No face detected in {img_path}")
                        continue
                        
                    # Get the face with highest confidence
                    face_obj = max(face_objs, key=lambda x: x.get("confidence", 0))
                    face_img = face_obj["face"]
                    
                    # Make sure face_img is in the right format for represent
                    if isinstance(face_img, np.ndarray):
                        if face_img.dtype != np.uint8:
                            face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
                    
                    # Generate embedding for this face
                    embedding_result = DeepFace.represent(
                        img_path=face_img,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        align=True
                    )
                    
                    # Extract the embedding vector
                    if isinstance(embedding_result, list) and len(embedding_result) > 0:
                        embedding = embedding_result[0].get("embedding", [])
                    else:
                        embedding = embedding_result.get("embedding", [])
                    
                    # Skip if embedding is empty
                    if not embedding:
                        continue
                    
                    # Store the embedding and image path
                    database[person_dir]["embeddings"].append(embedding)
                    database[person_dir]["image_paths"].append(img_path)
                    
                    print(f"Processed {img_path}")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        # Save the database as a JSON file
        db_path = f"{EMBEDDINGS_DIR}/DeepFace/{detector_backend.lower()}/{model_name.lower()}_db.json"
        
        # Convert numpy arrays to lists for JSON serialization
        for person_name in database:
            database[person_name]["embeddings"] = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding 
                                                 for embedding in database[person_name]["embeddings"]]
        
        with open(db_path, 'w') as f:
            json.dump(database, f)
        
        print(f"Database building complete at {db_path}")
        
    except Exception as e:
        print(f"Database building error: {e}")
        return
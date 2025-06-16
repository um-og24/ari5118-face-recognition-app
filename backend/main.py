import sys
sys.dont_write_bytecode = True

import cv2
import uvicorn
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from starlette.responses import JSONResponse

from services.system_monitoring_service import get_monitor

from utilities.configs import DATASET_DIR, UPLOADS_DIR, SUPPORTED_LIBRARIES, FACE_RECOGNITION_SUPPORTED_MODELS, DEEPFACE_SUPPORTED_BACKENDS_DETECTORS, DEEPFACE_SUPPORTED_MODELS, INSIGHTFACE_SUPPORTED_MODELS
from utilities.utils import purge_path, save_uploaded_file

from helpers.typed_classes import RecognitionResponse, FaceRecognitionResult, FaceRecognitionResult_Models

from libraries.face_recognition_library import recognize_faces_with_face_recognition, train_face_recognition_model
from libraries.deepface_library import recognize_faces_with_deepface, train_deepface_model
from libraries.insightface_library import train_insightface_model, recognize_faces_with_insightface

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = get_monitor()
monitor.start()

@app.get("/")
def root():
    return RedirectResponse(url="/ping")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/ping")
def health_check():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamp": timestamp, "status": "ok"}

@app.post("/train_model")
async def train_model(
    deepface_detector_model_pairs: Optional[str] = Form(None),
    facerecognition_models: Optional[str] = Form(None),
    insightface_models: Optional[str] = Form(None)
):
    try:
        facerecognition_model_list = [m.strip() for m in facerecognition_models.split(",")] if facerecognition_models else []
        insightface_model_list = [m.strip() for m in insightface_models.split(",")] if insightface_models else []

        deepface_pairs = []
        if deepface_detector_model_pairs:
            for pair in deepface_detector_model_pairs.split(","):
                if ":" not in pair:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace pair format: {pair}"})
                detector, model = pair.split(":")
                detector = detector.strip()
                model = model.strip()
                if detector not in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace detector: {detector}"})
                if model not in DEEPFACE_SUPPORTED_MODELS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace model: {model}"})
                deepface_pairs.append((detector, model))

        invalid_facerecognition = [m for m in facerecognition_model_list if m not in FACE_RECOGNITION_SUPPORTED_MODELS]
        invalid_insightface = [m for m in insightface_model_list if m not in INSIGHTFACE_SUPPORTED_MODELS]
        if invalid_facerecognition or invalid_insightface:
            errors = []
            if invalid_facerecognition:
                errors.append(f"Invalid FaceRecognition models: {', '.join(invalid_facerecognition)}")
            if invalid_insightface:
                errors.append(f"Invalid InsightFace models: {', '.join(invalid_insightface)}")
            return JSONResponse(status_code=422, content={"error": "; ".join(errors)})

        if not deepface_pairs and not facerecognition_model_list and not insightface_model_list:
            deepface_pairs = [(d, m) for d in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS for m in DEEPFACE_SUPPORTED_MODELS]
            facerecognition_model_list = FACE_RECOGNITION_SUPPORTED_MODELS
            insightface_model_list = INSIGHTFACE_SUPPORTED_MODELS

        monitor.reset()
        time.sleep(0.1)

        trained_models = []
        for model_name in insightface_model_list:
            train_insightface_model(DATASET_DIR, model_name)
            trained_models.append(f"InsightFace: {model_name}")

        for model_name in facerecognition_model_list:
            train_face_recognition_model(DATASET_DIR, model_name)
            trained_models.append(f"FaceRecognition: {model_name}")

        for detector_name, model_name in deepface_pairs:
            train_deepface_model(DATASET_DIR, detector_name, model_name)
            trained_models.append(f"DeepFace: {model_name} with {detector_name}")

        snapshot_end = monitor.get_snapshot()
        avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
        avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
        avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0

        return JSONResponse(status_code=200, content={
            "message": "Training completed!",
            "trained_models": trained_models,
            "avg_cpu_usage": avg_cpu,
            "avg_ram_usage": avg_ram
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/capture")
async def capture(file: UploadFile = File(...), library: str = Form(...), detector_name: str = Form(None), model_name: str = Form(...), name: str = Form("unknown_Person")):
    try:
        keep_a_raw_copy: bool = True

        info = await _parse_uploaded_image(file, keep_a_raw_copy)
        if "error" in info:
            return JSONResponse(status_code=500, content={"error": info["error"]})

        if library not in SUPPORTED_LIBRARIES:
            return JSONResponse(status_code=422, content={"error": f"This library ('{library}') is not supported!"})

        monitor.reset()
        time.sleep(0.1)

        if library == SUPPORTED_LIBRARIES[0]:
            if detector_name not in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                return JSONResponse(status_code=422, content={"error": f"This detector model ('{detector_name}') is not supported!"})
            if model_name not in DEEPFACE_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_deepface(info["rgb_image"], detector_name, model_name)
        elif library == SUPPORTED_LIBRARIES[1]:
            if model_name not in FACE_RECOGNITION_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_face_recognition(info["rgb_image"], model_name)
            detector_name = None
        elif library == SUPPORTED_LIBRARIES[2]:
            if model_name not in INSIGHTFACE_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_insightface(info["rgb_image"], model_name)
            detector_name = None

        if keep_a_raw_copy:
            save_uploaded_file(info["contents"], f"{UPLOADS_DIR}/{info['today']}/{name}", file.filename)
            if len(boxes) > 0:
                save_uploaded_file(info["contents"], f"{DATASET_DIR}/{name}", file.filename)
            purge_path(info["raw_file_path"])

        snapshot_end = monitor.get_snapshot()
        avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
        avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
        avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0

        return FaceRecognitionResult_Models(
            library=library,
            detector=detector_name,
            model=model_name,
            processing_time=timespan,
            boxes=boxes,
            names=names,
            avg_cpu_usage=avg_cpu,
            avg_ram_usage=avg_ram
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/detect")
async def detect(file: UploadFile = File(...), library: str = Form(...), detector_name: str = Form(None), model_name: str = Form(...)):
    try:
        keep_a_raw_copy: bool = True

        info = await _parse_uploaded_image(file, keep_a_raw_copy)
        if "error" in info:
            return JSONResponse(status_code=500, content={"error": info["error"]})

        if library not in SUPPORTED_LIBRARIES:
            return JSONResponse(status_code=422, content={"error": f"This library ('{library}') is not supported!"})

        monitor.reset()
        time.sleep(0.1)

        if library == SUPPORTED_LIBRARIES[0]:
            if detector_name not in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                return JSONResponse(status_code=422, content={"error": f"This detector model ('{detector_name}') is not supported!"})
            if model_name not in DEEPFACE_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_deepface(info["rgb_image"], detector_name, model_name)
        elif library == SUPPORTED_LIBRARIES[1]:
            if model_name not in FACE_RECOGNITION_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_face_recognition(info["rgb_image"], model_name)
            detector_name = None
        elif library == SUPPORTED_LIBRARIES[2]:
            if model_name not in INSIGHTFACE_SUPPORTED_MODELS:
                return JSONResponse(status_code=422, content={"error": f"This recognition model ('{model_name}') is not supported!"})
            boxes, names, timespan = recognize_faces_with_insightface(info["rgb_image"], model_name)
            detector_name = None

        if keep_a_raw_copy:
            for name in names:
                save_uploaded_file(info["contents"], f"{UPLOADS_DIR}/{info['today']}/{name}", file.filename)
            purge_path(info["raw_file_path"])

        snapshot_end = monitor.get_snapshot()
        avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
        avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
        avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0

        return FaceRecognitionResult_Models(
            library=library,
            detector=detector_name,
            model=model_name,
            processing_time=timespan,
            boxes=boxes,
            names=names,
            avg_cpu_usage=avg_cpu,
            avg_ram_usage=avg_ram
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/detect_and_compare", response_model=RecognitionResponse)
async def detect_and_compare(
    file: UploadFile = File(...),
    ground_truth: str = Form(None),
    deepface_detector_model_pairs: Optional[str] = Form(None),
    facerecognition_models: Optional[str] = Form(None),
    insightface_models: Optional[str] = Form(None)
):
    try:
        response: RecognitionResponse = RecognitionResponse(results=FaceRecognitionResult(models=[]))
        results: FaceRecognitionResult = response.results
        info = await _parse_uploaded_image(file)
        if "error" in info:
            return JSONResponse(status_code=500, content={"error": info["error"]})

        ground_truth_names = [name.strip() for name in ground_truth.split(",")] if ground_truth else ["unknown"]

        deepface_pairs = []
        if deepface_detector_model_pairs:
            for pair in deepface_detector_model_pairs.split(","):
                if ":" not in pair:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace pair format: {pair}"})
                detector, model = pair.split(":")
                detector = detector.strip()
                model = model.strip()
                if detector not in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace detector: {detector}"})
                if model not in DEEPFACE_SUPPORTED_MODELS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace model: {model}"})
                deepface_pairs.append((detector, model))

        facerecognition_model_list = [m.strip() for m in facerecognition_models.split(",")] if facerecognition_models else []
        insightface_model_list = [m.strip() for m in insightface_models.split(",")] if insightface_models else []

        invalid_facerecognition = [m for m in facerecognition_model_list if m not in FACE_RECOGNITION_SUPPORTED_MODELS]
        invalid_insightface = [m for m in insightface_model_list if m not in INSIGHTFACE_SUPPORTED_MODELS]
        if invalid_facerecognition or invalid_insightface:
            errors = []
            if invalid_facerecognition:
                errors.append(f"Invalid FaceRecognition models: {', '.join(invalid_facerecognition)}")
            if invalid_insightface:
                errors.append(f"Invalid InsightFace models: {', '.join(invalid_insightface)}")
            return JSONResponse(status_code=422, content={"error": "; ".join(errors)})

        if not deepface_pairs and not facerecognition_model_list and not insightface_model_list:
            deepface_pairs = [(d, m) for d in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS for m in DEEPFACE_SUPPORTED_MODELS]
            facerecognition_model_list = FACE_RECOGNITION_SUPPORTED_MODELS
            insightface_model_list = INSIGHTFACE_SUPPORTED_MODELS

        for detector_name, model_name in deepface_pairs:
            try:
                monitor.reset()
                time.sleep(0.1)
                boxes, names, timespan = recognize_faces_with_deepface(info["rgb_image"], detector_name, model_name)
                if boxes is None or names is None:
                    raise ValueError("Recognition failed: invalid boxes or names")
                accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                snapshot_end = monitor.get_snapshot()
                avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[0],
                    detector=detector_name,
                    model=model_name,
                    processing_time=timespan or 0.0,
                    boxes=boxes if boxes is not None else [],
                    names=names if names is not None else [],
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    avg_cpu_usage=avg_cpu,
                    avg_ram_usage=avg_ram
                ))
            except Exception as e:
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[0],
                    detector=detector_name,
                    model=model_name,
                    error_message=str(e)
                ))

        for model_name in facerecognition_model_list:
            try:
                monitor.reset()
                time.sleep(0.1)
                boxes, names, timespan = recognize_faces_with_face_recognition(info["rgb_image"], model_name)
                if boxes is None or names is None:
                    raise ValueError("Recognition failed: invalid boxes or names")
                accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                snapshot_end = monitor.get_snapshot()
                avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[1],
                    model=model_name,
                    processing_time=timespan or 0.0,
                    boxes=boxes if boxes is not None else [],
                    names=names if names is not None else [],
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    avg_cpu_usage=avg_cpu,
                    avg_ram_usage=avg_ram
                ))
            except Exception as e:
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[1],
                    model=model_name,
                    error_message=str(e)
                ))

        for model_name in insightface_model_list:
            try:
                monitor.reset()
                time.sleep(0.1)
                boxes, names, timespan = recognize_faces_with_insightface(info["rgb_image"], model_name)
                if boxes is None or names is None:
                    raise ValueError("Recognition failed: invalid boxes or names")
                accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                snapshot_end = monitor.get_snapshot()
                avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[2],
                    model=model_name,
                    processing_time=timespan or 0.0,
                    boxes=boxes if boxes is not None else [],
                    names=names if names is not None else [],
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    avg_cpu_usage=avg_cpu,
                    avg_ram_usage=avg_ram
                ))
            except Exception as e:
                results.models.append(FaceRecognitionResult_Models(
                    library=SUPPORTED_LIBRARIES[2],
                    model=model_name,
                    error_message=str(e)
                ))

        response.results = results
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/batch_benchmark", response_model=RecognitionResponse)
async def batch_benchmark(
    deepface_detector_model_pairs: Optional[str] = Form(None),
    facerecognition_models: Optional[str] = Form(None),
    insightface_models: Optional[str] = Form(None)
):
    try:
        response: RecognitionResponse = RecognitionResponse(results=FaceRecognitionResult(models=[]))
        results: FaceRecognitionResult = response.results
        test_json_path = "data/test_dataset.json"
        
        if not os.path.exists(test_json_path):
            return JSONResponse(status_code=404, content={"error": "test_dataset.json not found"})

        with open(test_json_path, 'r') as f:
            test_data = json.load(f)

        deepface_pairs = []
        if deepface_detector_model_pairs:
            for pair in deepface_detector_model_pairs.split(","):
                if ":" not in pair:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace pair format: {pair}"})
                detector, model = pair.split(":")
                detector = detector.strip()
                model = model.strip()
                if detector not in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace detector: {detector}"})
                if model not in DEEPFACE_SUPPORTED_MODELS:
                    return JSONResponse(status_code=422, content={"error": f"Invalid DeepFace model: {model}"})
                deepface_pairs.append((detector, model))

        facerecognition_model_list = [m.strip() for m in facerecognition_models.split(",")] if facerecognition_models else []
        insightface_model_list = [m.strip() for m in insightface_models.split(",")] if insightface_models else []

        invalid_facerecognition = [m for m in facerecognition_model_list if m not in FACE_RECOGNITION_SUPPORTED_MODELS]
        invalid_insightface = [m for m in insightface_model_list if m not in INSIGHTFACE_SUPPORTED_MODELS]
        if invalid_facerecognition or invalid_insightface:
            errors = []
            if invalid_facerecognition:
                errors.append(f"Invalid FaceRecognition models: {', '.join(invalid_facerecognition)}")
            if invalid_insightface:
                errors.append(f"Invalid InsightFace models: {', '.join(invalid_insightface)}")
            return JSONResponse(status_code=422, content={"error": "; ".join(errors)})

        if not deepface_pairs and not facerecognition_model_list and not insightface_model_list:
            deepface_pairs = [(d, m) for d in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS for m in DEEPFACE_SUPPORTED_MODELS]
            facerecognition_model_list = FACE_RECOGNITION_SUPPORTED_MODELS
            insightface_model_list = INSIGHTFACE_SUPPORTED_MODELS

        for test_image in test_data["test_images"]:
            image_path = test_image["path"]
            ground_truth_names = test_image["person_names"] if isinstance(test_image["person_names"], list) else [test_image["person_names"]]
            if not os.path.exists(image_path):
                continue

            with open(image_path, "rb") as f:
                contents = f.read()
            np_img = np.frombuffer(contents, np.uint8)
            rgb_image = cv2.cvtColor(cv2.imdecode(np_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            for detector_name, model_name in deepface_pairs:
                try:
                    monitor.reset()
                    time.sleep(0.1)
                    boxes, names, timespan = recognize_faces_with_deepface(rgb_image, detector_name, model_name)
                    if boxes is None or names is None:
                        raise ValueError("Recognition failed: invalid boxes or names")
                    accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                    snapshot_end = monitor.get_snapshot()
                    avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                    avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                    avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[0],
                        detector=detector_name,
                        model=model_name,
                        processing_time=timespan or 0.0,
                        boxes=boxes if boxes is not None else [],
                        names=names if names is not None else [],
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1_score,
                        avg_cpu_usage=avg_cpu,
                        avg_ram_usage=avg_ram,
                        image=test_image["filename"]
                    ))
                except Exception as e:
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[0],
                        detector=detector_name,
                        model=model_name,
                        error_message=str(e),
                        image=test_image["filename"]
                    ))

            for model_name in facerecognition_model_list:
                try:
                    monitor.reset()
                    time.sleep(0.1)
                    boxes, names, timespan = recognize_faces_with_face_recognition(rgb_image, model_name)
                    if boxes is None or names is None:
                        raise ValueError("Recognition failed: invalid boxes or names")
                    accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                    snapshot_end = monitor.get_snapshot()
                    avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                    avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                    avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[1],
                        model=model_name,
                        processing_time=timespan or 0.0,
                        boxes=boxes if boxes is not None else [],
                        names=names if names is not None else [],
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1_score,
                        avg_cpu_usage=avg_cpu,
                        avg_ram_usage=avg_ram,
                        image=test_image["filename"]
                    ))
                except Exception as e:
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[1],
                        model=model_name,
                        error_message=str(e),
                        image=test_image["filename"]
                    ))

            for model_name in insightface_model_list:
                try:
                    monitor.reset()
                    time.sleep(0.1)
                    boxes, names, timespan = recognize_faces_with_insightface(rgb_image, model_name)
                    if boxes is None or names is None:
                        raise ValueError("Recognition failed: invalid boxes or names")
                    accuracy, precision, recall, f1_score = _calculate_metrics(names, ground_truth_names, boxes)
                    snapshot_end = monitor.get_snapshot()
                    avg_cpu = [sum(core) / len(core) for core in snapshot_end.cpu] if snapshot_end.cpu else [0]
                    avg_cpu = sum(avg_cpu) / len(avg_cpu) if avg_cpu else 0
                    avg_ram = sum(snapshot_end.ram) / len(snapshot_end.ram) if snapshot_end.ram else 0
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[2],
                        model=model_name,
                        processing_time=timespan or 0.0,
                        boxes=boxes if boxes is not None else [],
                        names=names if names is not None else [],
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1_score,
                        avg_cpu_usage=avg_cpu,
                        avg_ram_usage=avg_ram,
                        image=test_image["filename"]
                    ))
                except Exception as e:
                    results.models.append(FaceRecognitionResult_Models(
                        library=SUPPORTED_LIBRARIES[2],
                        model=model_name,
                        error_message=str(e),
                        image=test_image["filename"]
                    ))

        response.results = results
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def _calculate_metrics(pred_names, ground_truth_names, boxes):
    """Calculate accuracy, precision, recall, and F1 score for face recognition with multiple ground truths."""
    if not boxes or boxes is None or pred_names is None or ground_truth_names is None:
        return 0.0, 0.0, 0.0, 0.0

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    matched_ground_truths = set()
    for pred_name in pred_names:
        if pred_name and pred_name in ground_truth_names and pred_name not in matched_ground_truths:
            TP += 1
            matched_ground_truths.add(pred_name)
        else:
            FP += 1

    FN = sum(1 for gt_name in ground_truth_names if gt_name and gt_name not in pred_names and gt_name != "unknown")
    
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return min(1.0, max(0.0, accuracy)), min(1.0, max(0.0, precision)), min(1.0, max(0.0, recall)), min(1.0, max(0.0, f1))

async def _parse_uploaded_image(image_file: UploadFile, keep_a_raw_copy: bool = False) -> dict:
    try:
        today = datetime.now().strftime("%Y%m%d")
        contents = await image_file.read()

        raw_file_path = None
        if keep_a_raw_copy:
            raw_file_path = save_uploaded_file(contents, f"{UPLOADS_DIR}/{today}/raw", image_file.filename)

        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return {"contents": contents, "rgb_image": rgb_image, "raw_file_path": raw_file_path, "today": today}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)



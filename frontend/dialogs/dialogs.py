import sys
sys.dont_write_bytecode = True

import streamlit as st
import requests

from utilities.configs import SUPPORTED_LIBRARIES, DEEPFACE_SUPPORTED_MODELS, FACE_RECOGNITION_SUPPORTED_MODELS, INSIGHTFACE_SUPPORTED_MODELS, DEEPFACE_SUPPORTED_BACKENDS_DETECTORS
from services.api_services import perform_retraining_of_models

@st.dialog("Confirm your action")
def confirm_delete(path: str):
    from services.io_services import delete_path
    st.write(f"Are you sure you want to delete?")
    col1, _, col2 = st.columns([1, 2, 1])
    with col1:
        if st.button("Yes", use_container_width=True):
            delete_path(path)
            st.rerun()
    with col2:
        if st.button("No", use_container_width=True):
            st.rerun()

@st.dialog("Confirm your action")
def confirm_retrain_models(selected_models: list = None, deepface_detectors: list = None):
    st.write(f"Are you sure you want to retrain the selected model(s)?")
    
    deepface_pairs = []
    facerecognition_models = []
    insightface_models = []
    model_to_library = {}
    
    for lib in SUPPORTED_LIBRARIES:
        models = {
            SUPPORTED_LIBRARIES[0]: DEEPFACE_SUPPORTED_MODELS,
            SUPPORTED_LIBRARIES[1]: FACE_RECOGNITION_SUPPORTED_MODELS,
            SUPPORTED_LIBRARIES[2]: INSIGHTFACE_SUPPORTED_MODELS
        }.get(lib, [])
        for model in models:
            model_to_library[f"{lib}: {model}"] = lib
    
    if selected_models:
        for model in selected_models:
            lib = model_to_library.get(model)
            model_name = model.split(": ")[1] if ": " in model else model
            if lib == SUPPORTED_LIBRARIES[0]:
                detectors = deepface_detectors if deepface_detectors else DEEPFACE_SUPPORTED_BACKENDS_DETECTORS
                for detector in detectors:
                    deepface_pairs.append(f"{detector}:{model_name}")
            elif lib == SUPPORTED_LIBRARIES[1]:
                facerecognition_models.append(model_name)
            elif lib == SUPPORTED_LIBRARIES[2]:
                insightface_models.append(model_name)
    else:
        for model in DEEPFACE_SUPPORTED_MODELS:
            for detector in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS:
                deepface_pairs.append(f"{detector}:{model}")
        facerecognition_models.extend(FACE_RECOGNITION_SUPPORTED_MODELS)
        insightface_models.extend(INSIGHTFACE_SUPPORTED_MODELS)
        
    selected = []
    if deepface_pairs:
        selected.append(f"DeepFace: {', '.join([pair.replace(':', ' with ') for pair in deepface_pairs])}")
    if facerecognition_models:
        selected.append(f"FaceRecognition: {', '.join(facerecognition_models)}")
    if insightface_models:
        selected.append(f"InsightFace: {', '.join(insightface_models)}")
    if selected:
        st.info(f"Selected models: {'; '.join(selected)}")
    else:
        st.info("No valid models selected. All models will be trained.")

    col1, _, col2 = st.columns([1, 2, 1])
    with col2:
        if st.button("No", use_container_width=True):
            st.rerun()
    with col1:
        retrain = st.button("Yes", use_container_width=True)

    if retrain:
        with st.spinner("Training model(s), please wait..."):
            try:
                result = perform_retraining_of_models(deepface_pairs, facerecognition_models, insightface_models)
                if "error" in result:
                    st.error(f"Training failed: {result['error']}")
                else:
                    st.success(f"Training completed! Trained models: {', '.join(result['trained_models'])}")
                    st.write(f"Average CPU Usage: {result['avg_cpu_usage']:.2f}%")
                    st.write(f"Average RAM Usage: {result['avg_ram_usage']:.2f}%")
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                st.rerun()
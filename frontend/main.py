import sys
sys.dont_write_bytecode = True

import streamlit as st
import streamlit.components.v1 as components

from utilities.configs import SUPPORTED_LIBRARIES, FACE_RECOGNITION_SUPPORTED_MODELS, DEEPFACE_SUPPORTED_BACKENDS_DETECTORS, DEEPFACE_SUPPORTED_MODELS, INSIGHTFACE_SUPPORTED_MODELS, LFW_DATASET_DIR
from ui_components.main_content_ui import main_contribution
from ui_components.gallery_ui import show_gallery
from ui_components.benchmarks_ui import run_benchmarks
from dialogs.dialogs import confirm_retrain_models

def init_session_state():
    # Initialize session state for batch results
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "batch_agg_results" not in st.session_state:
        st.session_state.batch_agg_results = None
    if "failed_images" not in st.session_state:
        st.session_state.failed_images = None

def main():
    st.set_page_config(page_title="Edge Face Recognition", layout="wide", initial_sidebar_state="collapsed")

    # Initialize session state
    init_session_state()

    st.title("Deep Learning for Computer Vision (ARI5118)")
    st.subheader("🔍 Real-Time Face Recognition on Edge Devices")

    sidebar()

    main_content()

def main_content():
    face_recognition_content, contribution, view_dataset, benchmarks, advanced_settings = st.tabs(["🔎 Face Recognition", "🤝🏻 Contribute", "📚 View Dataset", "📈 Run Benchmarks", "⚙️ Advanced Settings"])

    with face_recognition_content:
        st.subheader("Welcome To My Face Recognition App")
        st.write("Welcome to your private and secure face recognition platform, powered by lightweight AI models tailored for edge devices.")
        st.info("Select a face recognition library and model, then either upload an image containing faces or initiate a live camera feed, or provide an M3U8 or YouTube link to begin detection and recognition—all processed locally for maximum privacy and efficiency.")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Library")
                library = st.radio("Choose which library you wish to use:", options=SUPPORTED_LIBRARIES, horizontal=True)

            with col2:
                st.subheader("Model")
                detector_name = None
                model_name = None

                if library == SUPPORTED_LIBRARIES[0]:
                    detector_name = st.selectbox("Select preferred detector model:", DEEPFACE_SUPPORTED_BACKENDS_DETECTORS, disabled=False)
                    model_name = st.selectbox("Select preferred recognition model:", DEEPFACE_SUPPORTED_MODELS, disabled=False)
                elif library == SUPPORTED_LIBRARIES[1]:
                    model_name = st.selectbox("Select preferred recognition model:", FACE_RECOGNITION_SUPPORTED_MODELS, disabled=False)
                elif library == SUPPORTED_LIBRARIES[2]:
                    model_name = st.selectbox("Select preferred recognition model:", INSIGHTFACE_SUPPORTED_MODELS, disabled=False)

        main_contribution("detection", library, detector_name, model_name)

    with contribution:
        st.subheader("Disclaimer")
        st.info("""
        Your participation in this face detection/recognition project is strictly voluntary. Any facial data or related contributions you provide will be used exclusively for academic and research purposes only.

        All data collected will be handled with the highest level of confidentiality. Facial images, names, and any associated information will be securely stored and will not be shared outside the scope of this research.

        By participating, you acknowledge and consent to the use of your data under these conditions.
        """)

        st.warning("**Note:** Please do not forget to retrain the chosen model after your contribution by clicking on the '**Re-train Model**' button from the '⚙️ Advanced Settings' tab.")

        st.divider()

        name = st.text_input("Kindly enter your name:")
        if name.strip():
            main_contribution("contribution", library, detector_name, model_name, name.strip())
        else:
            st.error("Please enter a name before starting.")

    with view_dataset:
        st.subheader("Dataset Gallery")
        st.info("""
        Browse and manage all face contributions submitted so far.

        This gallery showcases every individual who has contributed and has been added to the system. Use this section to review or remove entries, and to ensure the recognition models remain up-to-date and relevant.

        (This section does not include the LFW dataset for performance reasons)
        """)

        # tabs=st.tabs(["LFW", "Custom (Contributions)"])
        # with tabs[0]:
        #     st.warning("Labelled Faces in the Wild dataset.")

        #     st.divider()

        #     show_gallery(dataset_dir=LFW_DATASET_DIR)

        # with tabs[1]:
        st.warning("This section is protected and only trusted authorized persons can access.")

        st.divider()

        show_gallery()

    with benchmarks:
        st.subheader("Benchmarks, Statistics & Comparisons")
        st.info("""
        This section provides access to performance data and comparative analysis across various models. Review standardized metrics, efficiency ratings, and capability assessments to better understand relative strengths and limitations. This benchmarking information helps guide selection decisions based on objective performance criteria.
        """)

        st.divider()

        run_benchmarks(library, detector_name, model_name)

    with advanced_settings:
        st.subheader("Retrain Libraries and Models")
        st.info("""
        Recreate embeddings for selected libraries and models using the current dataset. Leave selections empty to train all models.
        """)

        st.warning("**Note:** This operation is time-consuming. Monitor hardware performance in the sidebar.")

        st.divider()

        model_options = []
        model_to_library = {}
        for lib in SUPPORTED_LIBRARIES:
            models = {
                SUPPORTED_LIBRARIES[0]: DEEPFACE_SUPPORTED_MODELS,
                SUPPORTED_LIBRARIES[1]: FACE_RECOGNITION_SUPPORTED_MODELS,
                SUPPORTED_LIBRARIES[2]: INSIGHTFACE_SUPPORTED_MODELS
            }.get(lib, [])
            for model in models:
                display_name = f"{lib}: {model}"
                model_options.append(display_name)
                model_to_library[display_name] = lib

        selected_models = st.multiselect(
            "Select models to retrain (leave empty to train all models in all libraries)",
            model_options,
            default=[],
            placeholder="All models selected",
            key="train_models"
        )

        has_deepface = any(model_to_library.get(model) == SUPPORTED_LIBRARIES[0] for model in selected_models)

        deepface_detectors = []
        if has_deepface:
            with st.container():
                deepface_detectors = st.multiselect(
                    "Select DeepFace detectors for training (applies to DeepFace models only, leave empty for all detectors)",
                    DEEPFACE_SUPPORTED_BACKENDS_DETECTORS,
                    default=[],
                    placeholder="All detectors selected",
                    key="deepface_detectors_train"
                )

        if st.button("Re-train Models", use_container_width=True):
            confirm_retrain_models(selected_models, deepface_detectors)

    st.divider()

def sidebar() -> str:
    with st.sidebar:
        from utilities.configs import SYSTEM_MONITORING_URL
        components.iframe(SYSTEM_MONITORING_URL, height=1040, scrolling=False)

if __name__ == "__main__":
    try:
        main()
    finally:
        pass
import sys
sys.dont_write_bytecode = True

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from io import BytesIO

from fastapi.datastructures import UploadFile
from services import api_services, io_services
from ui_components.gallery_ui import center_an_image, plot_detection_results
from ui_components import image_file_uploader_ui, image_web_link_ui
from utilities.configs import DETECT_AND_COMPARE_API_URL, SUPPORTED_LIBRARIES, DEEPFACE_SUPPORTED_BACKENDS_DETECTORS, DEEPFACE_SUPPORTED_MODELS, FACE_RECOGNITION_SUPPORTED_MODELS, INSIGHTFACE_SUPPORTED_MODELS, TEST_DATASET_JSON_PATH, BENCHMARKS_REPORTS_DIR


def run_benchmarks(library: str, detector_name: str, model_name: str) -> None:
    st.subheader("Benchmark Options")
    st.info("Choose to benchmark using a predefined test dataset (test_dataset.json) or using a single image (via upload or web link). For single images, enter multiple ground truth names separated by commas if applicable.")

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
            model_to_library[model] = lib

    selected_models = st.multiselect(
        "Select models for benchmarking (leave empty for all models in all libraries)",
        model_options,
        default=[f"{library}: {model_name}"] if model_name and f"{library}: {model_name}" in model_options else [],
        placeholder="All models selected",
        key="models"
    )

    has_deepface = any(model_to_library.get(model) == SUPPORTED_LIBRARIES[0] for model in selected_models)

    deepface_detectors = []
    if has_deepface:
        with st.container():
            deepface_detectors = st.multiselect(
                "Select DeepFace detectors (applies to DeepFace models only, leave empty for all detectors)",
                DEEPFACE_SUPPORTED_BACKENDS_DETECTORS,
                default=[detector_name] if detector_name in DEEPFACE_SUPPORTED_BACKENDS_DETECTORS else [],
                placeholder="All detectors selected",
                key="deepface_detectors"
            )

    batch_test_tab, single_image_tab = st.tabs(["Batch Test", "Single Image"])

    with batch_test_tab:
        _run_batch_benchmarks(selected_models, deepface_detectors)

        if st.session_state.batch_results is not None and st.session_state.batch_agg_results is not None:
            st.divider()
            st.info("Displaying latest batch benchmark results.")
            if st.session_state.failed_images:
                st.warning(f"Failed to process {len(st.session_state.failed_images)} images: {', '.join(st.session_state.failed_images)}")
            _plot_metrics(st.session_state.batch_agg_results, is_batch=True)

    with single_image_tab:
        ground_truth_input = st.text_input("Ground Truth Names (comma-separated, e.g., 'John_Doe,Jane_Smith', default: unknown)", value="unknown", key="single_ground_truth")
        ground_truth = [name.strip() for name in ground_truth_input.split(",")] if ground_truth_input else ["unknown"]
        www_image_link_subtab, upload_image_subtab = st.tabs(["🌐 Web Image Link", "📤 Upload Image"])
        with www_image_link_subtab:
            _run_benchmarks_on_web_image_link(ground_truth, selected_models, deepface_detectors)
        with upload_image_subtab:
            _run_benchmarks_on_uploaded_image(ground_truth, selected_models, deepface_detectors)


def _run_benchmarks_on_web_image_link(ground_truth: list, selected_models: list, deepface_detectors: list) -> None:
    image_web_link_ui.show_component("benchmarks", DETECT_AND_COMPARE_API_URL, lambda api_url, image_bytes, image_file, data: _perform_and_plot_benchmarks_results(api_url, image_bytes, image_file, ground_truth=ground_truth, selected_models=selected_models, deepface_detectors=deepface_detectors))

def _run_benchmarks_on_uploaded_image(ground_truth: list, selected_models: list, deepface_detectors: list) -> None:
    image_file_uploader_ui.show_component("benchmarks", DETECT_AND_COMPARE_API_URL, lambda api_url, image_bytes, image_file, data: _perform_and_plot_benchmarks_results(api_url, image_bytes, image_file, ground_truth=ground_truth, selected_models=selected_models, deepface_detectors=deepface_detectors))


def _run_batch_benchmarks(selected_models: list, deepface_detectors: list) -> None:
    st.subheader("Batch Benchmarking")
    st.info(f"Run benchmarks on all images in test_dataset.json for selected models ({', '.join(selected_models) if selected_models else 'all'}). Results show average metrics.")

    test_data = io_services.read_json_content(TEST_DATASET_JSON_PATH)
    if test_data is None:
        st.error("'test_dataset.json' not found in data directory. Please create it with test image metadata.")
        return

    with st.expander("Preview Test Dataset Contents", expanded=False):
        _preview_test_dataset(test_data)

    if st.button("Run Batch Benchmark", use_container_width=True):
        progress_bar = st.progress(0.0, text="Processing images...")
        with st.spinner("Running batch benchmarks, please wait..."):
            try:
                results = []
                failed_images = []
                model_to_library = {f"{lib}: {model}": lib for lib in SUPPORTED_LIBRARIES for model in {
                    SUPPORTED_LIBRARIES[0]: DEEPFACE_SUPPORTED_MODELS,
                    SUPPORTED_LIBRARIES[1]: FACE_RECOGNITION_SUPPORTED_MODELS,
                    SUPPORTED_LIBRARIES[2]: INSIGHTFACE_SUPPORTED_MODELS
                }.get(lib, [])}

                deepface_pairs = []
                facerecognition_models = []
                insightface_models = []
                if selected_models:
                    for model_display in selected_models:
                        lib = model_to_library.get(model_display)
                        model_name = model_display.split(": ")[1] if ": " in model_display else model_display
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

                # Initialize progress bar
                total_images = len(test_data["test_images"])
                processed_images = 0
                progress_value = processed_images / total_images
                progress_bar.progress(progress_value, text=f"Processing image {processed_images + 1}/{total_images}")

                for test_image in test_data["test_images"]:
                    image_path = test_image["path"]
                    ground_truth = test_image["person_names"] if isinstance(test_image["person_names"], list) else [test_image["person_names"]]
                    if os.path.exists(image_path):
                        try:
                            with open(image_path, "rb") as f:
                                image_bytes = f.read()
                            image_file = UploadFile(filename=test_image["filename"], file=BytesIO(image_bytes))
                            data = {
                                "ground_truth": ",".join(ground_truth),
                                "deepface_detector_model_pairs": ",".join(deepface_pairs) if deepface_pairs else None,
                                "facerecognition_models": ",".join(facerecognition_models) if facerecognition_models else None,
                                "insightface_models": ",".join(insightface_models) if insightface_models else None
                            }
                            response_data = api_services.perform_face_recognition_task(DETECT_AND_COMPARE_API_URL, image_file, data)
                            if not isinstance(response_data, dict) or "results" not in response_data or not isinstance(response_data["results"], dict) or "models" not in response_data["results"]:
                                failed_images.append(f"{test_image['filename']} (Invalid response: {str(response_data)})")
                                continue
                            models = response_data["results"]["models"]
                            if not isinstance(models, list):
                                failed_images.append(f"{test_image['filename']} (Invalid models format: {str(models)})")
                                continue
                            for model in models:
                                if not isinstance(model, dict):
                                    failed_images.append(f"{test_image['filename']} (Invalid model data: {str(model)})")
                                if model["error_message"] or any(v is None for v in [model["accuracy"], model["precision"], model["recall"], model["f1_score"], model["processing_time"], model["avg_cpu_usage"], model["avg_ram_usage"]]):
                                    failed_images.append(f"{test_image['filename']} ({model.get('model', 'Unknown')}: {model.get('error_message', 'Missing metrics')})")
                                    continue
                                results.append({
                                    "Library": model["library"],
                                    "Detector": model["detector"] or "-",
                                    "Model": model["model"],
                                    "Accuracy (%)": model["accuracy"] * 100,
                                    "Precision (%)": model["precision"] * 100,
                                    "Recall (%)": model["recall"] * 100,
                                    "F1 Score (%)": model["f1_score"] * 100,
                                    "Processing Time (s)": model["processing_time"],
                                    "Avg CPU Usage (%)": model["avg_cpu_usage"],
                                    "Avg RAM Usage (%)": model["avg_ram_usage"],
                                    "Image": test_image["filename"]
                                })
                        except Exception as e:
                            failed_images.append(f"{test_image['filename']} ({str(e)})")
                    else:
                        failed_images.append(f"{test_image['filename']} (file not found)")

                    # Update progress bar
                    processed_images += 1
                    progress_value = processed_images / total_images
                    progress_bar.progress(progress_value, text=f"Processing image {processed_images}/{total_images}")

                # Clear progress bar after completion
                progress_bar.empty()

                if failed_images:
                    st.warning(f"Failed to process {len(failed_images)} images: {', '.join(failed_images)}")

                if not results:
                    st.warning("No valid results for the selected models. Ensure models are trained and test images are valid.")
                    return

                df = pd.DataFrame(results)
                max_time = df["Processing Time (s)"].max() or 1
                df["Normalized Processing Time (%)"] = 100 * (1 - df["Processing Time (s)"] / max_time) if max_time > 0 else 0
                
                agg_df = df.groupby(["Library", "Detector", "Model"]).agg({
                    "Accuracy (%)": "mean",
                    "Precision (%)": "mean",
                    "Recall (%)": "mean",
                    "F1 Score (%)": "mean",
                    "Processing Time (s)": "mean",
                    "Normalized Processing Time (%)": "mean",
                    "Avg CPU Usage (%)": "mean",
                    "Avg RAM Usage (%)": "mean"
                }).reset_index()

                # Store results in session state
                st.session_state.batch_results = df
                st.session_state.batch_agg_results = agg_df
                st.session_state.failed_images = failed_images

                df.to_csv(f"{BENCHMARKS_REPORTS_DIR}/full_batch_results.csv", index=False)
                agg_df.to_csv(f"{BENCHMARKS_REPORTS_DIR}/batch_results.csv", index=False)
                
                st.success("Batch benchmarking complete! Results saved to full_batch_results.csv and batch_results.csv.")

            except Exception as e:
                st.error(f"Error during batch benchmarking: {str(e)}")
                progress_bar.empty()

def _preview_test_dataset(test_data):
    try:
        if "test_images" in test_data and test_data["test_images"]:
            max_columns = 4
            columns = st.columns(max_columns)
            col_index = 0
            for i, img in enumerate(test_data["test_images"]):
                if col_index >= max_columns:
                    columns = st.columns(max_columns)
                    col_index = 0
                with columns[col_index].container(border=True):
                    cols=st.columns(2)
                    cols[0].subheader(f"Test Item {i + 1}")
                    if os.path.exists(img["path"]):
                        cols[0].image(img["path"], caption=img["filename"], width=150)
                    else:
                        cols[0].write(f"{img['filename']} (Not Found)")
                    cols[1].write("")
                    cols[1].write("**Ground Truth**")
                    cols[1].write("\n".join([f"- \"{item}\"" for item in img["person_names"]]))
                col_index += 1
        else:
            st.warning("No test images found in test_dataset.json.")
    except Exception as e:
        st.error(f"Error loading test_dataset.json: {str(e)}")


def _perform_and_plot_benchmarks_results(api_url: str, image_bytes: bytes, image_file: UploadFile, ground_truth: list, selected_models: list, deepface_detectors: list) -> None:
    model_to_library = {f"{lib}: {model}": lib for lib in SUPPORTED_LIBRARIES for model in {
        SUPPORTED_LIBRARIES[0]: DEEPFACE_SUPPORTED_MODELS,
        SUPPORTED_LIBRARIES[1]: FACE_RECOGNITION_SUPPORTED_MODELS,
        SUPPORTED_LIBRARIES[2]: INSIGHTFACE_SUPPORTED_MODELS
    }.get(lib, [])}

    deepface_pairs = []
    facerecognition_models = []
    insightface_models = []
    if selected_models:
        for model_display in selected_models:
            lib = model_to_library.get(model_display)
            model_name = model_display.split(": ")[1] if ": " in model_display else model_display
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

    data = {
        "ground_truth": ",".join(ground_truth),
        "deepface_detector_model_pairs": ",".join(deepface_pairs) if deepface_pairs else None,
        "facerecognition_models": ",".join(facerecognition_models) if facerecognition_models else None,
        "insightface_models": ",".join(insightface_models) if insightface_models else None
    }
    with st.spinner("Running benchmarks..."):
        response_data = api_services.perform_face_recognition_task(api_url, image_file, data)

    with st.spinner("Loading benchmarks..."):
        if "error" in response_data:
            st.error(f"ERROR: {response_data['error']}")
        else:
            _plot_benchmarks_results(image_bytes, response_data["results"]["models"])


def _plot_benchmarks_results(image_bytes: bytes, models) -> None:
    img = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    with st.expander("**Original Image**", expanded=True, icon="🖼️"):
        center_an_image(img[:, :, ::-1], channels="RGB")

    with st.expander("Detection Metrics", expanded=True, icon="📊"):
        _plot_metrics(models)

    st.subheader("Detection Results")
    max_columns = 4
    columns = st.columns(max_columns)
    col_index = 0
    for model in models:
        if col_index >= max_columns:
            columns = st.columns(max_columns)
            col_index = 0
        with columns[col_index]:
            plot_detection_results(img, model)
            col_index += 1

def _plot_metrics(models, is_batch: bool = False):
    metrics_data = []
    if isinstance(models, pd.DataFrame):
        for _, row in models.iterrows():
            metrics_data.append({
                "Library": row["Library"],
                "Detector": row["Detector"] or "-",
                "Model": row["Model"],
                "Accuracy (%)": row["Accuracy (%)"],
                "Precision (%)": row["Precision (%)"],
                "Recall (%)": row["Recall (%)"],
                "F1 Score (%)": row["F1 Score (%)"],
                "Processing Time (s)": row["Processing Time (s)"],
                "Normalized Processing Time (%)": row["Normalized Processing Time (%)"],
                "Avg CPU Usage (%)": row["Avg CPU Usage (%)"],
                "Avg RAM Usage (%)": row["Avg RAM Usage (%)"]
            })
    else:
        for model in models:
            if model["error_message"] is None:
                metrics_data.append({
                    "Library": model["library"],
                    "Detector": model["detector"] or "-",
                    "Model": model["model"],
                    "Accuracy (%)": model["accuracy"] * 100,
                    "Precision (%)": model["precision"] * 100,
                    "Recall (%)": model["recall"] * 100,
                    "F1 Score (%)": model["f1_score"] * 100,
                    "Processing Time (s)": model["processing_time"],
                    "Normalized Processing Time (%)": (100 * (1 - model["processing_time"] / max([m["processing_time"] for m in models if m["processing_time"]], default=1))),
                    "Avg CPU Usage (%)": model["avg_cpu_usage"],
                    "Avg RAM Usage (%)": model["avg_ram_usage"]
                })

    if not metrics_data:
        st.warning("No valid benchmark results available. Please ensure models are trained and the input is valid.")
        return

    df = pd.DataFrame(metrics_data)
    tabs = st.tabs(["Metrics", "Bar Chart", "Grouped Bar", "Radar Chart", "Heatmap", "Box Plot", "Scatter Plot", "Parallel Coordinates"])

    with tabs[0]:
        st.subheader("Metrics Table")
        st.dataframe(df, use_container_width=True)
        if is_batch:
            st.download_button(
                label="Download Batch Results as CSV",
                data=df.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv"
            )

    with tabs[1]:
        metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Normalized Processing Time (%)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        for metric in metrics_to_plot:
            df[f"{metric}_text"] = df.apply(lambda row: f"{row[metric]:.2f} ({row['Detector']})", axis=1)
            fig = px.bar(
                df,
                x="Model",
                y=metric,
                color="Library",
                text=f"{metric}_text",
                title=f"{metric} Comparison",
                hover_data=["Detector"]
            )
            fig.update_traces(textposition="auto")
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title=metric,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, key=f"{metric.replace(' ', '_').replace('(%)', '')}_bar_chart_is_batch_{is_batch}", use_container_width=True)
            df.drop(f"{metric}_text", axis=1, inplace=True)

    with tabs[2]:
        metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Normalized Processing Time (%)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=df["Model"],
                y=df[metric],
                name=metric,
                text=df[metric].apply(lambda x: f"{x:.2f}"),
                textposition="auto",
                hovertemplate="%{x}<br>%{y:.2f}<br>Library: " + df["Library"]
            ))
        fig.update_layout(
            barmode="group",
            title="Metric Comparison (Grouped Bar)",
            xaxis_title="Model",
            yaxis_title="Value",
            legend_title="Metric",
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, key=f"{metric.replace(' ', '_').replace('(%)', '')}_grouped_bar_chart_is_batch_{is_batch}", use_container_width=True)

    with tabs[3]:
        metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Normalized Processing Time (%)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill="toself",
                name=row["Model"],
                hovertemplate="%{theta}: %{r:.2f}<br>Library: " + row["Library"]
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Model Performance Comparison (Radar)",
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, key=f"radar_chart_is_batch_{is_batch}", use_container_width=True)

    with tabs[4]:
        metric_columns = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Normalized Processing Time (%)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        heatmap_data = df[["Model"] + metric_columns].set_index("Model")
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Metric", y="Model", color="Value (%)"),
            title="Performance Heatmap",
            color_continuous_scale="Viridis",
            text_auto=".2f",
            aspect="auto"
        )
        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title="Model",
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, key=f"performance_heatmap_is_batch_{is_batch}", use_container_width=True)

    with tabs[5]:
        st.subheader("Box Plot of Metrics")
        metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Processing Time (s)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        fig = go.Figure()
        for metric in metrics_to_plot:
            fig.add_trace(go.Box(
                y=df[metric],
                x=df["Model"],
                name=metric,
                hovertemplate="Model: %{x}<br>%{y:.2f}<br>Metric: " + metric
            ))
        fig.update_layout(
            title="Metric Distribution Across Models",
            xaxis_title="Model",
            yaxis_title="Value",
            height=600,
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig, key=f"box_plot_is_batch_{is_batch}", use_container_width=True)

    with tabs[6]:
        st.subheader("Scatter Plot of Accuracy vs. Processing Time")
        fig = px.scatter(
            df,
            x="Accuracy (%)",
            y="Processing Time (s)",
            color="Library",
            size="F1 Score (%)",
            hover_data=["Model", "Detector"],
            text="Model",
            title="Accuracy vs. Processing Time (Size: F1 Score)"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Accuracy (%)",
            yaxis_title="Processing Time (s)",
            height=600,
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig, key=f"scatter_plot_is_batch_{is_batch}", use_container_width=True)

    with tabs[7]:
        st.subheader("Parallel Coordinates Plot")
        metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Processing Time (s)", "Avg CPU Usage (%)", "Avg RAM Usage (%)"]
        fig = px.parallel_coordinates(
            df,
            color="Accuracy (%)",
            dimensions=metrics_to_plot,
            labels={m: m for m in metrics_to_plot},
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Parallel Coordinates of Metrics"
        )
        fig.update_layout(
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig, key=f"parallel_coordinates_plot_is_batch_{is_batch}", use_container_width=True)


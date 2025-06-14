import sys
sys.dont_write_bytecode = True

import streamlit as st

from services.io_services import assert_dir_path, exists, get_dirs, get_dir_content, delete_path, IMAGE_FILE_TYPES
from utilities.configs import DATASET_DIR, LFW_DATASET_DIR, GALLERY_MASTER_PASSWORD
from utilities.utils import extract_objects_from_image, get_a_random_number, get_an_empty_image
from dialogs.dialogs import confirm_delete

def show_gallery(person_name: str = "", correct_password: str = GALLERY_MASTER_PASSWORD, allow_removal: bool = False, dataset_dir: str = DATASET_DIR):
    assert_dir_path(dataset_dir)

    if not exists(dataset_dir):
        st.error("Nothing to display!")
        return

    persons_folders = get_dirs(dataset_dir)
    if len(persons_folders) == 0:
        st.error("No Content Available")
        return

    if dataset_dir == LFW_DATASET_DIR:
        inputted_password = GALLERY_MASTER_PASSWORD
    else:
        inputted_password = st.text_input("Password", type="password", key=f"gallery_password_{person_name}")
    if inputted_password != correct_password:
        st.error("Access Denied")
        return

    show_expanded = True
    if inputted_password == GALLERY_MASTER_PASSWORD:
        allow_removal = True
        show_expanded = False

    allow_removal = st.toggle("Enable Delete",value=allow_removal,key=f"allow_delete_{person_name}")

    persons_folders.sort()
    for person in persons_folders:
        if not person_name or person_name.lower() == person.lower():
            _display_content(person, show_expanded, allow_removal, dataset_dir)

def _display_content(person_name: str = "", show_expanded: bool = False, allow_removal: bool = False, dataset_dir: str = DATASET_DIR) -> None:
    person_images = get_dir_content(f"{dataset_dir}/{person_name}", IMAGE_FILE_TYPES)
    with st.expander(f"Gallery for '{person_name}' ({len(person_images)} images)", expanded=show_expanded):
        gallery_items_placeholder = st.container()

        st.write("---")

        if allow_removal:
            st.button(f"Forget about {person_name}", key=f"gallery_forget_about_{person_name}_{get_a_random_number(max=4500)}", use_container_width=True, on_click=confirm_delete, args=[f"{DATASET_DIR}/{person_name}"])

        with gallery_items_placeholder:
            _display_images(person_images, person_name, allow_removal=allow_removal, dataset_dir=dataset_dir)

def _display_images(images_names: list[str], person_name: str, max_columns: int = 4, allow_removal: bool = False, dataset_dir: str = DATASET_DIR) -> None:
    if len(images_names) > 0 and len(images_names) < max_columns:
        max_columns = len(images_names)

    st.write("---")
    columns=st.columns(max_columns)
    col_index=0
    for image_name in images_names:
        if col_index>=max_columns:
            st.write("---")
            columns=st.columns(max_columns)
            col_index=0

        with columns[col_index]:
            image_path = f"{dataset_dir}/{person_name}/{image_name}"

            if allow_removal:
                st.button("Remove", key=f"remove_{image_path}_{person_name}_{get_a_random_number(max=4500)}", use_container_width=True, on_click=confirm_delete, args=[image_path])

            if exists(image_path):
                center_an_image(image_path)
            else:
                center_an_image(get_an_empty_image(100, 100, "Not Found"))

            col_index+=1


def center_an_image(image, channels: str = "RGB", width: int = None , caption: str = None, use_container_width: bool = False) -> None:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        if width == None:
            use_container_width = True
            plot_an_image(image, channels=channels, width=width, caption=caption, use_container_width=use_container_width)


def plot_an_image(image, channels: str = "RGB", width: int = None , caption: str = None, use_container_width: bool = False) -> None:
    st.image(image, channels=channels, width=width, caption=caption, use_container_width=use_container_width)


def plot_detection_results(image, results, split_detected_face_in_columns: int = 4) -> None:
    boxes = results["boxes"]
    names = results["names"]

    expander_title=results["model"]
    if results["detector"]:
        expander_title=f"{results['detector']} - {results['model']}"
    with st.container(border=True):
        st.subheader(f"{results['library']} - {expander_title}")
        if results["error_message"]:
            st.error(f"**Error:** {results['error_message']}")
        else:
            if results["processing_time"]:
                st.write(f"**Time Elapsed:** {results['processing_time']:0.3f}s")

            if results["avg_cpu_usage"]:
                st.write(f"**Avg CPU Usage:** {results['avg_cpu_usage']:0.3f}%")

            if results["avg_ram_usage"]:
                st.write(f"**Avg RAM Usage:** {results['avg_ram_usage']:0.3f}%")

            st.write("**Face Names:**")
            st.write(names)
            # col1,col2=st.columns(2)
            # with col1:
            #     st.write("**Face Names:**")
            #     st.write(names)
            #     #st.write(names)
            # with col2:
            #     st.write("**Face Locations:**")
            #     st.write(boxes)
            #     #st.write(boxes)

        plot_detected_faces_only(image, boxes, names, split_detected_face_in_columns)

def plot_detected_faces_only(image, boxes, names, max_columns: int = 4) -> None:
    with st.expander("**Detected Face(s):**", expanded=False, icon="🕵🏻"):
        if len(names) == 0:
            image = get_an_empty_image(100, 100, "No Face")
            center_an_image(image)
            return

        if max_columns > 1:
            if len(names) < max_columns:
                max_columns = len(names)
            columns=st.columns(max_columns)
            col_index=0

        image = image.copy()

        faces = extract_objects_from_image(image, boxes, names)

        for face in faces:
            if max_columns > 1 and col_index>=max_columns:
                columns=st.columns(max_columns)
                col_index=0
            
            if max_columns > 1:
                with columns[col_index]:
                    face_img = face["object"]
                    name = face["text"]
                    plot_an_image(face_img[:, :, ::-1], caption=name)
                    col_index+=1
            else:
                face_img = face["object"]
                name = face["text"]
                plot_an_image(face_img[:, :, ::-1], caption=name)

import os

import streamlit as st

from ltt_ff_frontend.defect_review_ui import list_view


def app():
    st.title("Defect Review")

    col1, col2 = st.columns([1, 1])
    with col1:
        image_dir = st.text_input('Image Directory', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532')
    with col2:
        lrf_path = st.text_input('.lrf Path', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532_classified.lrf')

    if os.path.isdir(image_dir) and os.path.exists(lrf_path):
        list_view.app(image_dir, lrf_path)

    # # Select box (filter)
    # base_path = "/mnt/fs0/x9u_detection_result/"
    # folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f != "data_bk"]
    # filtered_folders = [f for f in folders if "classify" not in f]

    # # Parse URL to get the 'lot' parameter
    # query_params = st.query_params
    # lot_name = query_params.get('lot', None)

    # # Find the index of the lot_name in filtered_folders
    # if lot_name and lot_name in filtered_folders:
    #     selected_index = filtered_folders.index(lot_name)
    # else:
    #     selected_index = 0

    # selected_folder = st.selectbox("Select a lot", filtered_folders, index=selected_index, key="lot_selector")

    # if selected_folder:
    #     st.query_params.lot = selected_folder
    #     list_view.app(selected_folder)

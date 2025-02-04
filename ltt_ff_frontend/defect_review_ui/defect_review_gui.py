import os

import streamlit as st

from ltt_ff_frontend.defect_review_ui import list_view


def app():
    st.title("Defect Review")

    # Select box (filter)
    base_path = "/mnt/fs0/x9u_detection_result/"
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f != "data_bk"]
    filtered_folders = [f for f in folders if "classify" not in f]

    # Parse URL to get the 'lot' parameter
    query_params = st.query_params
    lot_name = query_params.get('lot', None)

    # Find the index of the lot_name in filtered_folders
    if lot_name and lot_name in filtered_folders:
        selected_index = filtered_folders.index(lot_name)
    else:
        selected_index = 0

    selected_folder = st.selectbox("Select a lot", filtered_folders, index=selected_index, key="lot_selector")

    if selected_folder:
        st.query_params.lot = selected_folder
        list_view.app(selected_folder)

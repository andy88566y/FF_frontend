import base64
import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review_ui import list_view
from ltt_ff_frontend.defect_ui.defect_ui_helper import check_matching_lot_id


def app():
    st.title("Defect Review")

    # Initialize session state for image_dir and lrf_path inputs
    if "image_dir" not in st.session_state:
        st.session_state.image_dir = ''
    if "lrf_path" not in st.session_state:
        st.session_state.lrf_path = ''

    # Parse URL to get the encoded 'image_dir' and 'lrf_path' parameters
    # They must have already been encoded using urlsafe_b64encode, then coverted to str.
    query_params = st.query_params
    encoded_image_dir_as_str = query_params.get('image_dir', None)
    encoded_lrf_path_as_str = query_params.get('lrf_path', None)

    # Decode image_dir and lrf_path if they are strings.
    # Decoding flow: encoded path as str > encoded path as bytes > decoded path as bytes > decoded path as str
    decoded_image_dir_as_str, decoded_lrf_path_as_str = '', ''
    if isinstance(encoded_image_dir_as_str, str):
        encoded_image_dir_as_bytes = str.encode(encoded_image_dir_as_str)
        decoded_image_dir_as_bytes = base64.urlsafe_b64decode(encoded_image_dir_as_bytes)
        decoded_image_dir_as_str = decoded_image_dir_as_bytes.decode()
        st.session_state.image_dir = decoded_image_dir_as_str
    if isinstance(encoded_lrf_path_as_str, str):
        encoded_lrf_path_as_bytes = str.encode(encoded_lrf_path_as_str)
        decoded_lrf_path_as_bytes = base64.urlsafe_b64decode(encoded_lrf_path_as_bytes)
        decoded_lrf_path_as_str = decoded_lrf_path_as_bytes.decode()
        st.session_state.lrf_path = decoded_lrf_path_as_str

    col1, col2 = st.columns([1, 1])
    with col1:
        text_input_image_dir = st.text_input(label='Image Directory', value=st.session_state.image_dir)
    with col2:
        text_input_lrf_path = st.text_input(label='.lrf Path', value=st.session_state.lrf_path)

    if not text_input_image_dir or not text_input_lrf_path:
        st.caption("Please input an Image Directory and an .lrf path to begin reviewing defects.")

    # Use the decoded image_dir and lrf_path if they are valid.
    # Otherwise, use the input from the text fields + encode them and store in query_params.
    if decoded_image_dir_as_str and decoded_lrf_path_as_str:
        if not os.path.isdir(decoded_image_dir_as_str):
            raise ValueError(f'Image directory in URL is invalid: {decoded_image_dir_as_str}')

        if not os.path.exists(decoded_lrf_path_as_str):
            raise ValueError(f'.lrf Path in URL is invalid: {decoded_lrf_path_as_str}')

        if not check_matching_lot_id(decoded_image_dir_as_str, decoded_lrf_path_as_str):
            raise ValueError(f'Lot IDs do not match: {decoded_image_dir_as_str} and {decoded_lrf_path_as_str}')

        logger.info('URL params successfully parsed.')
        list_view.app(decoded_image_dir_as_str, decoded_lrf_path_as_str)

    elif text_input_image_dir and text_input_lrf_path:
        if not os.path.isdir(text_input_image_dir):
            raise ValueError(f'Input Image directory in text field is invalid: {text_input_image_dir}')

        if not os.path.exists(text_input_lrf_path):
            raise ValueError(f'Input .lrf Path in text field is invalid: {text_input_lrf_path}')

        if not check_matching_lot_id(text_input_image_dir, text_input_lrf_path):
            raise ValueError(f'Lot IDs do not match: {text_input_image_dir} and {text_input_lrf_path}')

        st.query_params.image_dir = base64.urlsafe_b64encode(str.encode(text_input_image_dir)).decode()
        st.query_params.lrf_path = base64.urlsafe_b64encode(str.encode(text_input_lrf_path)).decode()
        logger.info('Input field params encoded and stored in URL.')
        list_view.app(text_input_image_dir, text_input_lrf_path)

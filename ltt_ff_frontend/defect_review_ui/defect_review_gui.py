import base64
import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_review_ui import list_view


def app() -> None:
    st.title("Defect Review")

    # TODO: Remove image_dir once our own image generation process is done
    # Initialize session state for result_dir and image_dir inputs
    if "result_dir" not in st.session_state:
        st.session_state.result_dir = ""
    if "image_dir" not in st.session_state:
        st.session_state.image_dir = ""

    # Parse URL to get the encoded 'result_dir' and 'image_dir' parameters
    # They must have already been encoded using urlsafe_b64encode, then coverted to str.
    encoded_result_dir_as_str = st.query_params.get("result_dir", None)
    encoded_image_dir_as_str = st.query_params.get("image_dir", None)

    # Decode result_dir and image_dir if they are strings.
    # Decoding flow: encoded path as str > encoded path as bytes > decoded path as bytes > decoded path as str
    decoded_result_dir_as_str, decoded_image_dir_as_str = "", ""
    if isinstance(encoded_result_dir_as_str, str):
        encoded_result_dir_as_bytes = str.encode(encoded_result_dir_as_str)
        decoded_result_dir_as_bytes = base64.urlsafe_b64decode(encoded_result_dir_as_bytes)
        decoded_result_dir_as_str = decoded_result_dir_as_bytes.decode()
    if isinstance(encoded_image_dir_as_str, str):
        encoded_image_dir_as_bytes = str.encode(encoded_image_dir_as_str)
        decoded_image_dir_as_bytes = base64.urlsafe_b64decode(encoded_image_dir_as_bytes)
        decoded_image_dir_as_str = decoded_image_dir_as_bytes.decode()

    # If text input fields are not empty, assign values to url params
    col1, col2 = st.columns([1, 1])
    with col1:
        text_input_result_dir = st.text_input(label="Result Directory", value=st.session_state.result_dir)
        if text_input_result_dir:
            text_input_result_dir = os.path.normpath(text_input_result_dir)
            st.query_params.result_dir = base64.urlsafe_b64encode(str.encode(text_input_result_dir)).decode()
    with col2:
        text_input_image_dir = st.text_input(label="Image Directory", value=st.session_state.image_dir)
        if text_input_image_dir:
            text_input_image_dir = os.path.normpath(text_input_image_dir)
            st.query_params.image_dir = base64.urlsafe_b64encode(str.encode(text_input_image_dir)).decode()

    if not text_input_result_dir or not text_input_image_dir:
        st.caption("Please input an Result Directory and Image Directory to begin reviewing defects.")

    # Use input from text fields if they exist
    # Otherwise, assign URL params to session state and refresh
    if text_input_result_dir and text_input_image_dir:
        if not os.path.isdir(text_input_result_dir):
            raise ValueError(f"Input Result directory in text field is invalid: {text_input_result_dir}")

        if not os.path.isdir(text_input_image_dir):
            raise ValueError(f"Input Image directory in text field is invalid: {text_input_image_dir}")

        logger.info("Input field params encoded and stored in URL.")
        list_view.app(text_input_result_dir, text_input_image_dir)

    # hotfix for endless rerun bug
    elif text_input_result_dir or text_input_image_dir:
        pass

    elif decoded_result_dir_as_str and decoded_image_dir_as_str:
        if not os.path.exists(decoded_result_dir_as_str):
            raise ValueError(f"Result directory in URL is invalid: {decoded_result_dir_as_str}")

        if not os.path.isdir(decoded_image_dir_as_str):
            raise ValueError(f"Image directory in URL is invalid: {decoded_image_dir_as_str}")

        logger.info("URL params successfully parsed.")
        st.session_state.result_dir = decoded_result_dir_as_str
        st.session_state.image_dir = decoded_image_dir_as_str
        st.rerun()

    else:
        pass

from datetime import date

import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, ModelArchitecture, ModelType, PixelSize, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.model_converter import convert_dualstream_cnn, convert_dualstream_cnn_v2
from ltt_ff_frontend.shared_components import helper


def app() -> None:
    #####################################################################################################
    # Model Conversion UI                                                                               #
    #####################################################################################################
    logger.debug("Loading Model Converter/Inspector UI...")
    st.title("Model Converter/Inspector")
    st.caption("Convert models from .ckpt to .pth or inspect model parameters")

    mode_select_col, model_type_select_col = st.columns(2)
    with mode_select_col:
        model_ui_mode = st.segmented_control(
            "Converter mode / Inspector mode", options=["Converter", "Inspector"], default="Converter"
        )
        if model_ui_mode is None:
            st.error("Please select a mode.")
            return

    if model_ui_mode == "Converter":
        with model_type_select_col:
            model_architecture = st.selectbox(
                label="Model Architecture", options=[ma.value for ma in ModelArchitecture]
            )

        if model_architecture == "DUALSTREAMCNN":
            convert_dualstream_cnn.app()
        elif model_architecture == "DUALSTREAMCNN_V2":
            convert_dualstream_cnn_v2.app()

    elif model_ui_mode == "Inspector":
        model_list = api_helper.get_base_models()
        model_to_inspect = st.selectbox(
            label="Model to inspect", options=model_list, format_func=helper.format_model_name
        )

        model_details = api_helper.get_model_details(model_to_inspect)
        if not model_details:
            st.error(f"Failed to get model details for {model_to_inspect}")
            return

        st.json(model_details)

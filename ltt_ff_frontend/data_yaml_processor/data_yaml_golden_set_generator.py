from datetime import datetime

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, MaskType, ModelType, PixelSize, Site, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.subheader(body="Generate a golden data set from a data yaml by relabeling and generating new LRFs.")

    data_yaml_path = st.text_input(label="Data Yaml Path", value="")

    output_dir_col, output_suffix_col = st.columns(2)
    with output_dir_col:
        output_dir = st.text_input(label="LRF Output Directory", value="")
    with output_suffix_col:
        output_suffix = st.text_input(label="Output LRF suffix", value="")

    if st.button(label="Generate golden set LRFs"):
        if not data_yaml_path or not output_dir:
            st.error("Please input Data Yaml Path, LRF Output Directory, and Output LRF suffix [optional].")
            logger.error("Please input Data Yaml Path, LRF Output Directory, and Output LRF suffix [optional].")
            return

        request = api_helper.generate_golden_set_from_data_yaml(
            data_yaml_path=data_yaml_path, output_dir=output_dir, output_suffix=output_suffix
        )

        if request.get("status") == "error":
            message = request.get("message")
            st.error(f"{message}")
            return

        st.success(f"{request.get('message')}")

    return

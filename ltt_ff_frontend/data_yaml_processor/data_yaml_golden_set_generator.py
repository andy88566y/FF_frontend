from datetime import datetime

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, MaskType, ModelType, PixelSize, Site, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.subheader(body="Generate a golden data set from a data yaml by relabeling and generating new LRFs.")

    data_yaml_file = st.file_uploader("Upload data yaml (.yaml)", type=".yaml")

    # Show preview of uploaded data yaml
    if data_yaml_file is not None:
        with st.expander(label="Data yaml preview"):
            data_yaml = yaml.load(data_yaml_file, Loader=yaml.Loader)
            st.json(data_yaml)

    output_dir = st.text_input(label="Output Directory", value="")

    output_prefix_col, output_suffix_col = st.columns(2)
    with output_prefix_col:
        output_prefix = st.text_input(label="Output LRF prefix", value="", placeholder="Optional")
    with output_suffix_col:
        output_suffix = st.text_input(label="Output LRF suffix", value="", placeholder="Optional")

    # Automatically grab missed defects by reading result .db?
    auto_grab_missed_defects = st.toggle(label="Automatically grab missed defects from inference results", value=False)
    if auto_grab_missed_defects:
        result_dir = st.text_input(
            label="Inference Result Directory",
            help="To automatically retrieve missed defects, access to .db files is required",
        )
    else:
        result_dir = ""

    copy_images = st.toggle(
        label="Copy Image Directory to Output Directory and generate updated Data Yaml",
        value=False,
    )

    if copy_images:
        st.text(
            "If enabled, image_dir must be included in the data yaml. If there are missing image_dir entries, "
            + "they will not be copied, and the list of lots with missing image_dir will be shown."
        )
        st.text(
            "A new data yaml file with the new LRF path and new image_dir will be generated in the LRF Output "
            + "Directory. It can be directly  used as a data yaml file on the Inference UI."
        )

        remove_existing_image_dir = st.toggle(label="Remove existing Image Directory in Output Directory", value=False)
    else:
        remove_existing_image_dir = False

    if st.button(label="Generate golden set LRFs"):
        if not data_yaml_file or not output_dir:
            st.error("Please input Data Yaml Path and LRF Output Directory.")
            logger.error("Please input Data Yaml Path and LRF Output Directory.")
            return

        request = api_helper.generate_golden_set_from_data_yaml(
            data_yaml=data_yaml,
            output_dir=output_dir,
            output_prefix=output_prefix,
            output_suffix=output_suffix,
            copy_images=copy_images,
            remove_existing_image_dir=remove_existing_image_dir,
            inference_result_dir=result_dir,
        )

        if request.get("status") == "error":
            message = request.get("message")
            st.error(f"{message}")
            return

        lot_ids_without_image_dir = request.get("lot_ids_without_image_dir", [])
        if len(lot_ids_without_image_dir) > 0:
            st.warning(
                "Lots without image_dir were found. For the following lots, golden LRF was generated, but"
                + " image_dir was not copied to LRF Output Directory, and it is not included in the golden yaml.\n\n"
                + f"{lot_ids_without_image_dir}"
            )

        st.success(f"{request.get('message')}")

    return

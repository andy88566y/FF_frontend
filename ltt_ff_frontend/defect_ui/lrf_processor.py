import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import INFERENCE_DEFAULT_RESULT_DIR
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper


def app() -> None:
    logger.debug("Loading LRF Processor...")
    st.title("LRF Processor")
    st.caption("Split, merge, and filter LRF files!")

    mode = st.segmented_control(label="Mode", options=["Split", "Merge", "Filter"], default="Split")
    if mode is None:
        st.error("Please select a mode!")
        return

    if mode == "Split":
        lrf_path_col, partitions_col = st.columns([5, 1])
        with lrf_path_col:
            lrf_path = st.text_input(
                label="LRF path",
                value="",
            )

        with partitions_col:
            partitions = st.number_input(label="Number of partitions", min_value=2, max_value=10, value=2, step=1)

        output_dir = st.text_input(
            label="Output directory for split LRF files (must be empty)",
            value="",
        )

        if st.button(label="Split LRF", type="primary"):
            if not lrf_path or not output_dir:
                st.error("Please fill in both the LRF path and output directory.")
                return

            if not os.path.exists(lrf_path):
                st.error(f"LRF path does not exist: {lrf_path}")
                return

            if os.path.exists(output_dir) and os.listdir(output_dir):
                st.error(f"Output directory is not empty: {output_dir}")
                return

            os.makedirs(name=output_dir, exist_ok=True)

            request = api_helper.split_lrf(lrf_path=lrf_path, partitions=partitions, output_dir=output_dir)

            if request.json().get("status") == "error":
                message = request.json().get("message")
                st.error(f"{message}")
            else:
                st.success(f"Successfully split LRF into {partitions} parts at {output_dir}.")

    elif mode == "Merge":
        output_dir = st.text_input(
            label="Directory containing split LRF files",
            value="",
        )

        files_found = os.listdir(output_dir)
        files_found.sort()

        if files_found:
            st.text(f"Files found in {output_dir}:")
            for file in files_found:
                st.text(file)
            st.warning("LRF partition files will be deleted after merging!")
        else:
            st.error(f"No files found in {output_dir}")
            return

        if st.button(label="Merge LRF", type="primary"):
            request = api_helper.merge_lrf(output_dir=output_dir)

            if request.json().get("status") == "error":
                message = request.json().get("message")
                st.error(f"{message}")
            else:
                st.success(f"Successfully merged LRF to {output_dir}.")

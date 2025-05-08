import os
from enum import Enum

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


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
            partitions = st.number_input(label="Number of partitions", min_value=2, value=2, step=1)

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

        if not output_dir:
            st.error("Please enter Directory containing split LRF files")
            return

        files_found = os.listdir(output_dir)
        try:
            files_found = sorted(files_found, key=lambda x: int(x.split("_")[0]))
        except ValueError as e:
            st.error(f"Invalid file name found: {str(e)}.")
            logger.error(f"Invalid file name found: {str(e)}.")
            return

        if files_found:
            st.text(f"Files found in {output_dir}:")
            st.json(files_found)
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

    elif mode == "Filter":

        class FilterMode(Enum):
            KEEP = "Keep the defects"
            REMOVE = "Remove the defects"

        lrf_path = st.text_input(
            label="LRF path",
            value="",
        )

        output_dir = st.text_input(
            label="Output directory for filtered LRF",
            value="",
        )

        csv_uploader_col, keep_remove_toggle_col, _ = st.columns([1, 1, 1])
        with csv_uploader_col:
            csv_file = st.file_uploader("Upload filter config (.csv)", type="csv")
        with keep_remove_toggle_col:
            keep_items_in_filter = st.segmented_control(
                "Keep or remove the defects in the filter?",
                options=[FilterMode.KEEP.value, FilterMode.REMOVE.value],
                default=FilterMode.KEEP.value,
                help=(
                    "E.g. defect list = [1, 2, 3, 4, 5], filter = [1, 2, 6]  \n"
                    "Keep the defects results: [1, 2]  \n"
                    "Remove the defects results: [3, 4, 5]  \n"
                ),
            )
            if keep_items_in_filter is None:
                st.error("Please select to keep the defects or remove the defects.")
                return

        # Convert csv to df
        # Remove unneeded columns / rows to reduce data trf during API call
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            required_columns = {"No", "UniqueID"}

            # Keep only "No" and "UniqueID" columns
            trimmed_df = df[[col for col in df.columns if any(val in required_columns for val in df[col])]]

            # Set first row values as column names and remove first 2 rows
            trimmed_df.columns = pd.Index(trimmed_df.iloc[0])
            trimmed_df = trimmed_df[2:].reset_index(drop=True)

        if not lrf_path or not csv_file:
            st.warning("Please enter LRF path and upload a filter config file.")
            return

        if st.button("Generate filtered LRF", type="primary"):
            request = api_helper.filter_lrf(
                lrf_path=lrf_path,
                output_dir=output_dir,
                keep_defects_in_filter=True if keep_items_in_filter == FilterMode.KEEP.value else False,
                defect_filters=trimmed_df.to_json(),
            )

            if request.json().get("status") == "error":
                message = request.json().get("message")
                st.error(f"{message}")
            else:
                st.success(f"{request.json().get('message')}")

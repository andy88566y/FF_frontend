import os
from enum import Enum

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
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

    if not lrf_path or not output_dir or not csv_file:
        st.warning("Please enter LRF path, Output Directory, and upload a filter config file.")
        return

    # Convert csv to df
    # Remove unneeded columns / rows to reduce data trf during API call
    df = pd.read_csv(csv_file)
    required_columns = {"No", "UniqueID"}

    # Keep only "No" and "UniqueID" columns
    trimmed_df = df[[col for col in df.columns if any(val in required_columns for val in df[col])]]

    # Set first row values as column names and remove first 2 rows
    trimmed_df.columns = pd.Index(trimmed_df.iloc[0])
    trimmed_df = trimmed_df[2:].reset_index(drop=True)

    with st.expander(label="Filter Preview"):
        st.dataframe(data=trimmed_df, hide_index=True)

    if st.button("Generate filtered LRF", type="primary"):
        if not os.path.exists(lrf_path):
            st.error(f"LRF path does not exist: {lrf_path}")
            return

        os.makedirs(name=output_dir, exist_ok=True)

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

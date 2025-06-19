import os
from typing import Literal

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    lrf_path = st.text_input(
        label="LRF path",
        value="",
    )
    output_dir = st.text_input(
        label="Output directory for re-labeled LRF",
        value="",
    )

    relabel_mode_col, relabel_csv_col = st.columns(2)
    with relabel_mode_col:
        relabel_mode: Literal["No/UniqueID", "ClassType"] | None = st.segmented_control(
            label="Re-label by No/UniqueID or by ClassType",
            options=["No/UniqueID", "ClassType"],
            default="No/UniqueID",
        )
        if relabel_mode is None:
            st.warning("Please select a re-label mode.")
            return

    with relabel_csv_col:
        csv_file = st.file_uploader("Upload re-label config (.csv)", type="csv")

    with st.expander(label=".csv example"):
        if relabel_mode == "No/UniqueID":
            example_df = pd.DataFrame(
                {
                    "No": [1, 3, 5, 7, 9],
                    "ClassType": [0, 0, 0, 9, 9],
                }
            )
            st.dataframe(data=example_df, hide_index=True, use_container_width=False)
            st.caption(body="Re-label defects no. 1, 3, 5 to ClassType 0, and defects no. 7, 9 to ClassType 9.")
        elif relabel_mode == "ClassType":
            example_df = pd.DataFrame(
                {
                    "Current_ClassType": [1, 2, 3, 4],
                    "New_ClassType": [0, 0, 9, 9],
                }
            )
            st.dataframe(data=example_df, hide_index=True, use_container_width=False)
            st.caption(
                body="Re-label all defects with ClassType 1, 2 to ClassType 0, and all defects with ClassType 3, 4 "
                + "to ClassType 9."
            )

    if not lrf_path or not output_dir or not csv_file:
        st.warning("Please enter LRF path, Output Directory, and upload a re-label config file.")
        return

    relabel_df = pd.read_csv(csv_file)

    with st.expander(label="Re-label Config Preview"):
        st.dataframe(data=relabel_df, hide_index=True, use_container_width=False)

    if st.button(label="Re-label LRF", type="primary"):
        if not os.path.exists(lrf_path):
            st.error(f"LRF path does not exist: {lrf_path}")
            return

        os.makedirs(name=output_dir, exist_ok=True)

        relabel_dict = dict(zip(relabel_df[relabel_df.columns[0]], relabel_df[relabel_df.columns[1]]))

        request = api_helper.relabel_lrf(
            lrf_path=lrf_path,
            output_dir=output_dir,
            relabel_mode=relabel_mode,
            relabel_map=relabel_dict,
        )

        if request.json().get("status") == "error":
            message = request.json().get("message")
            st.error(f"{message}")
        else:
            st.success(f"{request.json().get('message')}")

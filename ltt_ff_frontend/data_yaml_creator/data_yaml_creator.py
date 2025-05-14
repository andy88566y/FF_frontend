from datetime import datetime

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    logger.debug("Loading Data Yaml Creator...")
    st.title("Data Yaml Creator")
    st.caption("Create a data yaml file from LRFList csv file!")

    lrf_list_csv = st.file_uploader("Upload LRFList (.csv)", type="csv")

    st.subheader(
        body="Replace Windows path with Linux path", help="e.g. E:/AAA/BBB will be converted to /mnt/dbpc/AAA/BBB"
    )
    original_col, replace_col = st.columns(2)
    with original_col:
        original_str = st.text_input(label="Original path", value="E:")
    with replace_col:
        replace_str = st.text_input(label="Replacement path", value="/mnt/dbpc")

    if lrf_list_csv is None or not original_str or not replace_str:
        st.error("Please upload a LRFList, and enter an Original path and a Replacement path.")
        return

    df = pd.read_csv(lrf_list_csv)

    with st.expander(label="LRFList Preview"):
        st.dataframe(data=df)

    # Remove unneeded data from df
    required_columns = ["FileName", "MaskName1", "Insp.StartTime", "FilePath"]

    # Keep only "No" and "UniqueID" columns
    trimmed_df = df[required_columns]

    if st.button(label="Prepare data yaml", type="primary"):
        request = api_helper.lrflist_to_yaml(
            df=trimmed_df,
            original_str=original_str,
            replace_str=replace_str,
        )

        if request.get("status") == "error":
            message = request.get("message")
            st.error(f"{message}")
            return

        st.success(f"{request.get('message')}")
        data_yaml = request["data_yaml"]

        st.download_button(
            label="Download data yaml file",
            data=data_yaml,
            file_name=f"data_yaml_{datetime.now().astimezone()}.yaml",
            mime="text/yaml",
        )

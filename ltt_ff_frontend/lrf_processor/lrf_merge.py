import os

import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    output_dir = st.text_input(
        label="Directory containing split LRF files",
        value="",
    )

    remove_partitions = st.toggle(label="Remove partition LRF files after merging", value=False)

    if not output_dir:
        st.error("Please enter Directory containing split LRF files")
        return

    files_found = os.listdir(output_dir)
    try:
        files_found = sorted(files_found, key=lambda x: int(x.split("_")[0]))
    except ValueError as e:
        st.error(f"Invalid file name found: {str(e)}.  \nReminder: Directory must only contain split LRF files.")
        logger.error(f"Invalid file name found: {str(e)}.  \nReminder: Directory must only contain split LRF files.")
        return

    if files_found:
        st.text(f"Files found in {output_dir}:")
        st.json(files_found)
    else:
        st.error(f"No files found in {output_dir}")
        return

    if remove_partitions:
        st.warning("The files listed above will be deleted after merging!")

    if st.button(label="Merge LRF", type="primary"):
        request = api_helper.merge_lrf(output_dir=output_dir, remove_partitions=remove_partitions)

        if request.json().get("status") == "error":
            message = request.json().get("message")
            st.error(f"{message}")
        else:
            st.success(f"Successfully merged LRF to {output_dir}.")

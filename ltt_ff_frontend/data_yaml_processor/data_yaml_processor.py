import streamlit as st
from loguru import logger

from ltt_ff_frontend.data_yaml_processor import (
    data_yaml_compiler,
    data_yaml_creator,
    data_yaml_filter,
    data_yaml_golden_set_generator,
    data_yaml_parser,
)


def app() -> None:
    logger.debug("Loading Data Yaml Processor...")
    st.title("Data Yaml Processor")
    st.caption("Various utility tools for data yaml files!")

    processor_mode = st.segmented_control(
        label="Mode", options=["Create", "Filter", "Parser", "Golden Set Generator", "Compiler"], default="Create"
    )
    if processor_mode is None:
        st.error("Please select a mode!")
        return

    if processor_mode == "Create":
        data_yaml_creator.app()
    elif processor_mode == "Filter":
        data_yaml_filter.app()
    elif processor_mode == "Parser":
        data_yaml_parser.app()
    elif processor_mode == "Golden Set Generator":
        data_yaml_golden_set_generator.app()
    elif processor_mode == "Compiler":
        data_yaml_compiler.app()

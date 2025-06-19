import os
from enum import Enum
from typing import Literal

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.lrf_processor import lrf_filter, lrf_merge, lrf_relabel, lrf_split


def app() -> None:
    logger.debug("Loading LRF Processor...")
    st.title("LRF Processor")
    st.caption("Split, merge, filter, and re-label LRF files!")

    mode = st.segmented_control(label="Mode", options=["Split", "Merge", "Filter", "Re-label"], default="Split")
    if mode is None:
        st.error("Please select a mode!")
        return

    if mode == "Split":
        lrf_split.app()
    elif mode == "Merge":
        lrf_merge.app()
    elif mode == "Filter":
        lrf_filter.app()
    elif mode == "Re-label":
        lrf_relabel.app()

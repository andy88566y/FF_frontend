import base64
import os
from pprint import pformat
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT, BLANK_MODEL, INFERENCE_DEFAULT_RESULT_DIR, RESTRICT_OUTPUT_DIR, TIMEOUT
from ltt_ff_frontend.shared_components.helper import format_model_name
from ltt_ff_frontend.helpers import api_helper

def gen(df: pd.DataFrame) -> None:
    render_cols = st.columns(len(df.columns) + 1, vertical_alignment="top")
    for job_id, render_col in zip(df.columns, render_cols[1:]):
        with render_col:
            cannot_stop = df.loc["status", job_id] in ["completed", "error", "stopped", "stopping"]
            if st.button(f"stop {job_id}", key=f"stop-{job_id}", use_container_width=True, disabled=cannot_stop):
                message = api_helper.request_stop_job(job_id)
                st.write(message)
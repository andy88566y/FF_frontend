import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper


def gen(df: pd.DataFrame) -> None:
    render_cols = st.columns(len(df.columns) + 1, vertical_alignment="top")
    for job_id, render_col in zip(df.columns, render_cols[1:]):
        with render_col:
            cannot_stop = df.loc["status", job_id] in ["completed", "error", "stopped", "stopping"]
            if st.button(f"stop {job_id}", key=f"stop-{job_id}", use_container_width=True, disabled=cannot_stop):
                message = api_helper.request_stop_job(job_id)
                st.write(message)

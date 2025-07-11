from typing import Any, Literal

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


"""
Shows a summary of OOS lots in a result directory.
The format of this summary mirrors the run test excel table done by the PE team.
"""


def gen(oos_calculation: dict[str, Any]) -> None:
    all_summary = oos_calculation["all"]
    greater_equal_150_summary = oos_calculation[">=150"]
    smaller_150_summary = oos_calculation["<150"]
    particle_mode_count = oos_calculation["particle_mode_count"]

    st.caption(f"As-is defects contains {particle_mode_count} particle mode defects.")

    df = pd.DataFrame([all_summary, greater_equal_150_summary, smaller_150_summary])
    df = df[
        [
            "Mode",
            "LOT",
            "As-is",
            "To-be",
            "AFD",
            "MDC",
            "FF",
            "CR%",
            "FFR%",
            "FFR per Lots",
            "A. Miss Catch",
            "B. High False",
            "Success Lots",
        ]
    ]
    st.dataframe(data=df, hide_index=True, use_container_width=False)

from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def gen(
    recipe: dict[str, Any],
    inference_result_dir: str,
) -> None:
    missed_defects = api_helper.get_missed_defects(recipe, inference_result_dir)
    df = pd.DataFrame(columns=["Lot ID", "Missed defects"])
    for missed_defect_per_lot in missed_defects:
        for k, v in missed_defect_per_lot.items():
            df.loc[len(df)] = pd.Series({"Lot ID": k, "Missed defects": v if len(v) > 0 else ["No missed defects!"]})
    st.dataframe(df)

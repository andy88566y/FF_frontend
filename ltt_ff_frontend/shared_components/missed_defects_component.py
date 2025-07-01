from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def gen(missed_defects: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(columns=["Lot ID", "Missed defects"])

    for missed_defect_per_lot in missed_defects:
        for k, v in missed_defect_per_lot.items():
            df.loc[len(df)] = pd.Series({"Lot ID": k, "Missed defects": v if len(v) > 0 else ["No missed defects!"]})

    st.download_button(
        label="Download missed defect list as csv",
        data=df.to_csv(index=False),
        file_name=f"inference_result_{datetime.now().astimezone()}.csv",
        mime="text/csv",
    )

    st.dataframe(df)

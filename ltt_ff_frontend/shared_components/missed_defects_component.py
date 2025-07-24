from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


def gen(missed_defect_info: dict[str, Any]) -> None:
    df = pd.DataFrame(columns=["Lot ID", "Missed defects", "LRF Path"])

    missed_defect_lists = missed_defect_info["missed_defect_list"]
    lrf_paths = missed_defect_info["lrf_path_list"]

    for missed_defect_per_lot, lrf_path in zip(missed_defect_lists, lrf_paths):
        for k, v in missed_defect_per_lot.items():
            df.loc[len(df)] = pd.Series({"Lot ID": k, "Missed defects": v, "LRF Path": lrf_path})

    st.download_button(
        label="Download missed defect list as csv",
        data=df.to_csv(index=False),
        file_name=f"inference_result_{datetime.now().astimezone()}.csv",
        mime="text/csv",
    )

    st.dataframe(df)

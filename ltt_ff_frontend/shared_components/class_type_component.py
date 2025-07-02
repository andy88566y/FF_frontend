from typing import Any

import pandas as pd
import streamlit as st


def gen(classtype_count_list: list[dict[str, Any]]) -> None:
    for classtype_count in classtype_count_list:
        for lot_id, (lrf_type, counts) in classtype_count.items():
            st.text(f"Lot ID: {lot_id}")
            st.caption(f"LRF type: {lrf_type}")

            classtype_counter_df = pd.DataFrame.from_records(data=counts).sort_values(
                by="ClassType", ascending=True, key=lambda classtype: classtype.astype(int)
            )

            # Reset index after sorting, and let it start from 1 instead of 0
            classtype_counter_df.reset_index(inplace=True, drop=True)
            classtype_counter_df.index = range(1, len(classtype_counter_df) + 1)

            # Rearrage columns
            classtype_counter_df = classtype_counter_df[["Classification", "ClassType", "Count"]]

            st.dataframe(data=classtype_counter_df)

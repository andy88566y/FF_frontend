import pandas as pd
import streamlit as st

from ltt_ff_frontend.shared_components.chart_drawer.upset import plot_upset


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
    sort_by: str,
    exclude_zero: bool,
) -> tuple:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, tp_corrections, tn_corretions = intersections

    tp_df = pd.DataFrame(tp_corrections, columns=model_names)
    tn_df = pd.DataFrame(tn_corretions, columns=model_names)

    tp_col, tn_col = st.columns(2)
    with tp_col:
        if not (tp_df == 0).all().all():
            st.plotly_chart(
                plot_upset(
                    dataframes=[tp_df],
                    exclude_zeros=exclude_zero,
                    legendgroups=["True Defect"],
                    sorted_x=sort_by if sort_by != "default" else None,
                    sorted_y=sort_by if sort_by != "default" else None,
                    title=f"True Defects Upset Chart (Total: {ans.count(1)})",
                )
            )
        else:
            st.markdown("No True Defects Were Predicted Correct!")
    with tn_col:
        if not (tn_df == 0).all().all():
            st.plotly_chart(
                plot_upset(
                    dataframes=[tn_df],
                    exclude_zeros=exclude_zero,
                    legendgroups=["False Defects"],
                    sorted_x=sort_by if sort_by != "default" else None,
                    sorted_y=sort_by if sort_by != "default" else None,
                    title=f"False Defects Upset Chart (Total: {ans.count(0)})",
                )
            )
        else:
            st.markdown("No False Defects Were Predicted Correct!")

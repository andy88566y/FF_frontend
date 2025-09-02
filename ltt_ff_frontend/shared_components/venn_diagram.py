import numpy as np
import streamlit as st

from ltt_ff_frontend.shared_components.chart_drawer.venn import plot_venn


def gen(
    aggregated_model_data: tuple[list[str], list[dict], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> None:
    _, _, _, ans, _ = aggregated_model_data
    tp_datas, tn_datas, tp_corrections, tn_corrections = tuple(intersections)
    tp_sets = [set(model_correction) for model_correction in tp_datas]
    tn_sets = [set(model_correction) for model_correction in tn_datas]
    missed_tp_count = np.sum(np.all(np.asarray(tp_corrections) == 0, axis=1))
    missed_tn_count = np.sum(np.all(np.asarray(tn_corrections) == 0, axis=1))

    tp_col, tn_col = st.columns(2)
    with tp_col:
        st.plotly_chart(
            plot_venn(
                sets=tp_sets,
                labels=model_names,
                miss_count=missed_tp_count,
                title=f"True Defects Venn Diagram (Total: {ans.count(1)})",
            ),
            clear_figure=True,
        )
    with tn_col:
        st.plotly_chart(
            plot_venn(
                sets=tn_sets,
                labels=model_names,
                miss_count=missed_tn_count,
                title=f"False Defects Venn Diagram (Total: {ans.count(0)})",
            ),
            clear_figure=True,
        )

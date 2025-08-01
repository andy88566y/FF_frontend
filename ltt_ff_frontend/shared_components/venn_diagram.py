import streamlit as st
from ltt_ff_frontend.shared_components.chart_drawer.venn import plot_venn


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> None:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, _, _ = intersections
    tp_sets = [set(model_correction) for model_correction in tp_datas]
    tn_sets = [set(model_correction) for model_correction in tn_datas]

    tp_col, tn_col = st.columns(2)
    with tp_col:
        st.plotly_chart(
            plot_venn(sets=tp_sets, labels=model_names, title=f"True Defects Venn Diagram (Total: {ans.count(1)})"),
            clear_figure=True,
        )
    with tn_col:
        st.plotly_chart(
            plot_venn(sets=tn_sets, labels=model_names, title=f"False Defects Venn Diagram (Total: {ans.count(0)})"),
            clear_figure=True,
        )

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm


COLORS = ("#0072B2", "#D55E00", "#009E73")


def get_venn_fig(sets, labels, title):
    fig = plt.figure(figsize=(8, 6))
    venn3(
        sets,
        set_labels=labels,
        set_colors=COLORS,
        layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1)),
    )
    fig.tight_layout()
    fig.suptitle(title)
    return plt.gcf()


def gen(
    aggregated_model_data: tuple[list[str], list[float], list[int], list[str]],
    intersections: tuple[list[list[int]]],
    model_names: list[str],
    split_lot: bool,
) -> None:
    defect_ids, prob_lists, ans, lot_ids = aggregated_model_data
    classifications = ["Defect" if label == 1 else "Non-defect" if label == 0 else "Unlabeled" for label in ans]
    tp_datas, tn_datas, _ = intersections
    tp_sets = [set(model_correction) for model_correction in tp_datas]
    tn_sets = [set(model_correction) for model_correction in tn_datas]

    tp_col, tn_col = st.columns(2)
    with tp_col:
        st.pyplot(get_venn_fig(tp_sets, model_names, f"True Defects Venn Diagram (Total: {ans.count(1)})"))
    with tn_col:
        st.pyplot(get_venn_fig(tn_sets, model_names, f"None Defects Venn Diagram (Total: {ans.count(0)})"))

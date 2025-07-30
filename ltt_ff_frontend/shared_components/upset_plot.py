import matplotlib.pyplot as plt
import streamlit as st
from upsetplot import UpSet, from_memberships


def convert_to_set(model_names: list[str], corrections: list[list[int]]) -> list[list[str]]:
    model_sets = []
    for correction in corrections:
        if correction:
            model_sets.append([model_names[i] for i, c in enumerate(correction) if c])
    return model_sets


def get_upset_fig(data: list[list[str]], sort_by: str, exclude_zero: bool, title: str):
    memberships = from_memberships(data)
    # Generate and display the plot
    fig = plt.figure(figsize=(8, 6))
    UpSet(
        memberships, show_counts=True, subset_size="count", sort_by=sort_by, include_empty_subsets=not exclude_zero
    ).plot(fig)
    fig.tight_layout()
    fig.suptitle(title)

    return plt.gcf()


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
    tp_set = convert_to_set(model_names, tp_corrections)
    tn_set = convert_to_set(model_names, tn_corretions)
    tp_col, tn_col = st.columns(2)
    with tp_col:
        if any(len(tp) != 0 for tp in tp_set):
            st.pyplot(
                get_upset_fig(
                    data=tp_set,
                    sort_by=sort_by,
                    exclude_zero=exclude_zero,
                    title=f"True Defects Upset Chart (total: {ans.count(1)})",
                )
            )
        else:
            st.markdown("No True Defects Were Predicted Correct!")
    with tn_col:
        if any(len(tn) != 0 for tn in tn_set):
            st.pyplot(
                get_upset_fig(
                    data=tn_set,
                    sort_by=sort_by,
                    exclude_zero=exclude_zero,
                    title=f"False Defects Upset Chart (total: {ans.count(0)})",
                )
            )
        else:
            st.markdown("No False Defects Were Predicted Correct!")

from typing import Any

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers.api_helper import MultiLotModelData


def highlight_capture_rate(column):
    styles = []
    for val in column:
        numeric_val = float(val)
        color = "red" if numeric_val != 1.0 and numeric_val != -1.0 else ""
        font_weight = "bold" if numeric_val != 1.0 and numeric_val != -1.0 else ""
        styles.append(f"color: {color}; font-weight: {font_weight};")
    return styles

def highlight_to_be_total_defect_count(row):
    numeric_total_to_be_count = float(row[("Total Defect Count", "To-be")])
    numeric_true_defect_count = float(row[("True Defect Count", "Before")])

    row_styles = [""] * len(row)
    if numeric_total_to_be_count > 150 and numeric_true_defect_count <= 150:
        index = row.index.get_loc(("Total Defect Count", "To-be"))
        row_styles[index] = "color: red; font-weight: bold;"
    else:
        pass
    return row_styles

def draw_column_background_color(s):
    colors = {
        "red": "background-color: #ffcccb",
        "yellow": "background-color: #ffeb3b",
        "green": "background-color: #d4edda",
        "blue": "background-color: #d1ecf1",
        "default": "background-color: #d3d3d3",
    }
    col_to_colors = {
        "Total Defect Count": colors["yellow"],
        "True Defect Count": colors["red"],
        "Non Defect Count": colors["green"],
        "Unlabeled Count": colors["blue"],
    }
    foo =  [col_to_colors.get(first_index, colors["default"]) for first_index in s.index.get_level_values(0)]
    return foo

def calculate_filtered_results(
    raw_data: tuple[list[int], list[float], list[int]], selected_threshold: float
) -> dict[str, Any]:
    _, probability_list, answer_list = raw_data

    positive = answer_list.count(1)
    negative = answer_list.count(0)
    unlabeled = answer_list.count(-1)
    true_positive = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == 1
    )
    false_positive = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == 0
    )
    true_negative = sum(1 for prob, ans in zip(probability_list, answer_list) if prob < selected_threshold and ans == 0)
    filtered_unlabeled_defect_count = sum(
        1 for prob, ans in zip(probability_list, answer_list) if prob >= selected_threshold and ans == -1
    )

    as_is_defect_count = positive + negative + unlabeled
    to_be_defect_count = true_positive + false_positive + filtered_unlabeled_defect_count

    capture_rate = true_positive / positive if positive > 0 else -1
    capture_rate = true_positive / positive if positive > 0 else -1
    false_filter_rate = true_negative / negative if negative > 0 else -1
    filter_rate = 1 - (to_be_defect_count / as_is_defect_count) if as_is_defect_count > 0 else -1

    return {
        "as_is_defect_count": as_is_defect_count,
        "to_be_defect_count": to_be_defect_count,
        "filter_rate": filter_rate,
        "as_is_true_defect_count": positive,
        "to_be_true_defect_count": true_positive,
        "capture_rate": capture_rate,
        "as_is_non_defect_count": negative,
        "to_be_non_defect_count": false_positive,
        "false_filter_rate": false_filter_rate,
        "unlabeled": unlabeled,
        "filtered_unlabeled_defect_count": filtered_unlabeled_defect_count,
    }

def draw_stats_df(
    multilot_model_data: MultiLotModelData,
    selected_threshold,
    key: str,
) -> list[str]:
    model_metadata_list, defect_id_lists, probability_lists, answer_lists = (
        multilot_model_data.model_metadata_list,
        multilot_model_data.defect_id_lists,
        multilot_model_data.probability_lists,
        multilot_model_data.answer_lists
    )

    data = []
    for id_list, prob_list, ans_list, meta in zip(
        defect_id_lists,
        probability_lists,
        answer_lists,
        model_metadata_list
    ):
        count_rate_data = calculate_filtered_results((id_list, prob_list, ans_list), selected_threshold)
        data.append(
            [
                meta["lot_id"],
                count_rate_data["as_is_defect_count"],
                count_rate_data["to_be_defect_count"],
                f"{count_rate_data['filter_rate']:.4f}",
                count_rate_data["as_is_true_defect_count"],
                count_rate_data["to_be_true_defect_count"],
                f"{count_rate_data['capture_rate']:.4f}",
                count_rate_data["as_is_non_defect_count"],
                count_rate_data["to_be_non_defect_count"],
                f"{count_rate_data['false_filter_rate']:.4f}",
                count_rate_data["unlabeled"],
                count_rate_data["filtered_unlabeled_defect_count"],
            ]
        )
    return gen_stats_df_by_data_list(data, key)

def gen_stats_df_by_data_list(
    data: list[list[str]],
    key: str,
) -> list[str]:
    index = [
        ("Lot", "ID"),
        ("Total Defect Count", "As-is"),
        ("Total Defect Count", "To-be"),
        ("Total Defect Count", "Filter Rate"),
        ("True Defect Count", "Before"),
        ("True Defect Count", "After"),
        ("True Defect Count", "Capture Rate"),
        ("Non Defect Count", "Before"),
        ("Non Defect Count", "After"),
        ("Non Defect Count", "False Filter Rate"),
        ("Unlabeled Count", "Total"),
        ("Unlabeled Count", "Filtered"),
    ]
    pd_multiindex = pd.MultiIndex.from_tuples(index)
    df = pd.DataFrame(data, columns=pd_multiindex)
    styled_df = (
        df.style.apply(draw_column_background_color, axis=1)
        .apply(highlight_to_be_total_defect_count, axis=1)
        .apply(highlight_capture_rate, subset=[("True Defect Count", "Capture Rate")], axis=0)
    )
    event = st.dataframe(
        styled_df, key=key, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row"
    )

    selected_rows = event.selection.rows
    selected_df = df.iloc[selected_rows]
    selected_lot_id_list = selected_df["Lot"]["ID"].tolist()
    return selected_lot_id_list

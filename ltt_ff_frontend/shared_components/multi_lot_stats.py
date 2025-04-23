from typing import Any

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper
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
    foo = [col_to_colors.get(first_index, colors["default"]) for first_index in s.index.get_level_values(0)]
    return foo


def draw_stats_df(
    multi_lot_model_data: MultiLotModelData,
    recipe: dict[str, Any],
    inference_result_dir: str,
    key: str,
) -> list[str]:
    rows = []
    count_rate_data = api_helper.calculate_recipe_filtered_results(inference_result_dir, recipe=recipe)
    for data, meta in zip(count_rate_data, multi_lot_model_data.model_metadata_list):
        rows.append(
            [
                meta["lot_id"],
                data["as_is_defect_count"],
                data["to_be_defect_count"],
                f"{data['filter_rate']:.4f}",
                data["as_is_true_defect_count"],
                data["to_be_true_defect_count"],
                f"{data['capture_rate']:.4f}",
                data["as_is_non_defect_count"],
                data["to_be_non_defect_count"],
                f"{data['false_filter_rate']:.4f}",
                data["unlabeled"],
                data["filtered_unlabeled_defect_count"],
            ]
        )
    return gen_stats_df_by_data_list(rows, key)


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
        styled_df,
        key=key,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=35 * (len(data) + 2),
    )

    selected_rows = event.selection.rows
    selected_df = df.iloc[selected_rows]
    selected_lot_id_list = selected_df["Lot"]["ID"].tolist()
    return selected_lot_id_list

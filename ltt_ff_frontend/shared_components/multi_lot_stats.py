from typing import Any

import pandas as pd
import streamlit as st


BG_COLORS = {
    "red": "#ffcccb",
    "yellow": "#ffeb3b",
    "green": "#d4edda",
    "blue": "#d1ecf1",
    "default": "#d3d3d3",
}

TEXT_COLORS = {
    "red": "#FF0000",
    "orange": "#FF8C00",
    "yellow": "#FFDE59",
    "green": "#008000",
    "blue": "#1109D4",
    "default": "#000000",
}


def highlight_capture_rate(column):
    styles = []
    for val in column:
        numeric_val = float(val)
        text_color = TEXT_COLORS["red"] if numeric_val != 1.0 and numeric_val != -1.0 else ""
        font_weight = "bold" if numeric_val != 1.0 and numeric_val != -1.0 else ""
        styles.append(f"color: {text_color}; font-weight: {font_weight};")
    return styles


def highlight_oos(row):
    """
    Highlight colors for metrics table

    FFR OOS(Out Of Spec) definition
    - Defect > 150
        - FFR > 90% (BLUE) -> Pass
        - FFR < 90% (RED) -> OOS
    - Defect <= 150
        - To-Be <= 20 (GREEN) -> Pass
        - 20 < To-Be <= 150
            - FFR >= 90% (GREEN) -> Pass
            - FFR < 90% (ORANGE) -> OOS, but not as critical
        - To-Be > 150 (RED) -> OOS
    """
    true_defect_before_count_key = ("True Defect Count", "Before")
    false_filter_rate_key = ("Non Defect Count", "False Filter Rate")
    total_defect_to_be_count_key = ("Total Defect Count", "To-be")
    true_defect_count = float(row[true_defect_before_count_key])
    false_filter_rate = float(row[false_filter_rate_key])
    total_defect_to_be_count = float(row[total_defect_to_be_count_key])

    to_be_defect_count_index = row.index.get_loc(total_defect_to_be_count_key)
    false_filter_rate_index = row.index.get_loc(false_filter_rate_key)

    if false_filter_rate == -1:
        highlight_text_color = TEXT_COLORS["default"]
    else:
        if true_defect_count > 150:
            if false_filter_rate >= 0.9:
                # blue
                highlight_text_color = TEXT_COLORS["blue"]
            else:
                # red
                highlight_text_color = TEXT_COLORS["red"]
        else:
            if total_defect_to_be_count <= 20:
                # green
                highlight_text_color = TEXT_COLORS["default"]
            elif 20 < total_defect_to_be_count and total_defect_to_be_count <= 150:
                if false_filter_rate >= 0.9:
                    # green
                    highlight_text_color = TEXT_COLORS["default"]
                elif false_filter_rate < 0.9:
                    # orange, OOS, but not as critical
                    highlight_text_color = TEXT_COLORS["orange"]
            elif total_defect_to_be_count > 150:
                # red, OOS
                highlight_text_color = TEXT_COLORS["red"]
    row_styles = [""] * len(row)
    row_styles[to_be_defect_count_index] = f"color: {highlight_text_color}; font-weight: bold;"
    row_styles[false_filter_rate_index] = f"color: {highlight_text_color}; font-weight: bold;"

    return row_styles


def draw_column_background_color(s):
    bg_css_settings = {k: f"background-color: {v}" for k, v in BG_COLORS.items()}
    col_to_colors = {
        "Total Defect Count": bg_css_settings["yellow"],
        "True Defect Count": bg_css_settings["red"],
        "Non Defect Count": bg_css_settings["green"],
        "Unlabeled Count": bg_css_settings["blue"],
    }
    return [col_to_colors.get(first_col, bg_css_settings["default"]) for first_col in s.index.get_level_values(0)]


def gen(
    inference_data: list[list[Any]],
    key: str,
) -> tuple[list[str], pd.DataFrame]:
    rows = inference_data

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
    df = pd.DataFrame(rows, columns=pd_multiindex)

    styled_df = (
        df.style.apply(draw_column_background_color, axis=1)
        .apply(highlight_oos, axis=1)
        .apply(highlight_capture_rate, subset=[("True Defect Count", "Capture Rate")], axis=0)
    )
    event = st.dataframe(
        styled_df,
        key=key,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=35 * (len(rows) + 2),
    )

    selected_rows = event.selection.rows
    selected_df = df.iloc[selected_rows]
    selected_lot_id_list = selected_df["Lot"]["ID"].tolist()
    return selected_lot_id_list, df

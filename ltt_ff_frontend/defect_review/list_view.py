import operator
import random
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from loguru import logger


def hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)] + [160]


# Function to generate a list of colors for a given number of clusters
def generate_colors(num_clusters: int) -> list[str]:
    random.seed(num_clusters)
    # Use the 'viridis' colormap which can handle a large number of distinct colors
    cmap = plt.get_cmap("hsv", num_clusters)
    # Generate the colors
    colors = [mpl.colors.to_hex(cmap(i)) for i in range(num_clusters)]
    # Shuffle the colors
    random.shuffle(colors)
    return colors


def app(result_dir: str, selected_lot_id: str, models_name: list, lrf_ext: str, df: pd.DataFrame) -> None:
    st.session_state.filtered_df = df
    # Initialize session state for selected index
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = 0
    if "selected_map_index" not in st.session_state:
        st.session_state.selected_map_index = 0

    # Initialize session state for selection source
    if "selection_source" not in st.session_state:
        st.session_state.selection_source = ""

    # Initialize session state for filter criteria
    if "filter_column" not in st.session_state:
        st.session_state.filter_column = df.columns[0]
    if "filter_operator" not in st.session_state:
        st.session_state.filter_operator = "="
    if "filter_value" not in st.session_state:
        st.session_state.filter_value = ""
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = df

    # Reset session state values when changing folder
    previous_result_dir = st.session_state.get("result_dir", None)
    previous_lot_id = st.session_state.get("lot_id", None)
    if previous_result_dir != result_dir or previous_lot_id != selected_lot_id:
        st.session_state.filter_column = df.columns[0]
        st.session_state.filter_operator = "="
        st.session_state.filter_value = ""
        st.session_state.filtered_df = df
        st.session_state.selection_source = ""
        st.session_state.result_dir = result_dir
        st.session_state.lot_id = selected_lot_id

    filter_options_col, filter_operators_col, filter_value_col, apply_col, remove_col = st.columns([1, 1, 2, 1, 1])

    # Add filter options
    with filter_options_col:
        # Define the columns I want to display
        if result_dir:
            specific_columns = ["UniqueID", "X", "Y", "ClassType", "GT", "Pred", "Cluster"]
            for model_name in models_name:
                specific_columns.extend("P_" + model_name)
        else:
            if lrf_ext == "lrf":
                specific_columns = ["X", "Y", "ClassType", "GT", "Cluster"]
            else:
                specific_columns = ["UniqueID", "X", "Y", "ClassType", "GT", "Cluster"]
        if models_name:
            for model_name in models_name:
                specific_columns.append("P_" + model_name)

        # Filter the DataFrame columns to only include the specific columns
        filtered_columns = [col for col in df.columns if col in specific_columns]
        # Use the filtered columns in the selectbox
        st.session_state.filter_column = st.selectbox(
            label="Filter options",
            options=filtered_columns,
            index=filtered_columns.index(st.session_state.filter_column),
        )

    ops = {
        "=": operator.eq,
        ">": operator.gt,
        ">=": operator.ge,
        "<=": operator.le,
        "<": operator.lt,
    }

    with filter_operators_col:
        operator_columns = list(ops.keys())
        index = (
            operator_columns.index(st.session_state.filter_operator)
            if st.session_state.filter_operator in operator_columns
            else 0
        )
        st.session_state.filter_operator = st.selectbox(
            label="Filter operators",
            options=operator_columns,
            index=index,
        )

    # Add filter value text input, confirm button, and cancel button
    with filter_value_col:
        st.session_state.filter_value = st.text_input(
            label="Filter value",
            value=st.session_state.filter_value,
        )
    with apply_col:
        if st.button(label="Apply Filter", use_container_width=True):
            try:
                col = st.session_state.filter_column
                op = ops[st.session_state.filter_operator]
                val = st.session_state.filter_value

                # Try to convert value to the same type as the column
                col_dtype = df[col].dtype
                if col_dtype.kind in "iuf":  # numeric types
                    val = float(val)
                elif col_dtype.kind == "b":  # boolean
                    val = val.lower() in ["true", "1", "yes"]
                # else keep as string
                st.session_state.filtered_df = df[op(df[col], val)]
            except Exception as e:
                st.warning(f"Could not apply filter: {e}")

    with remove_col:
        if st.button(label="Remove Filter", icon=":material/close:", use_container_width=True):
            st.session_state.filter_column = df.columns[0]
            st.session_state.filter_value = ""
            st.session_state.filtered_df = df
            st.rerun()

    # List view
    st.subheader("List View")

    # Select only the columns I want to display
    if result_dir:
        all_columns = ["No", "UniqueID", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Pred", "P_rank", "Cluster"]
    else:
        if lrf_ext == "lrf":
            all_columns = ["No", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Cluster"]
        else:
            all_columns = ["No", "UniqueID", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Cluster"]

    if models_name is not None:
        for model_name in models_name:
            all_columns.append("P_" + model_name)
    # Sample DataFrame (replace with your actual data)
    df = st.session_state.filtered_df
    # Let user select columns to display
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=all_columns,
        default=all_columns,  # You can change this to a subset if needed
    )

    # Display the selected columns
    if selected_columns:
        st.session_state.filtered_df = df[selected_columns]
    else:
        st.warning("Please select at least one column to display.")

    listview_df = st.session_state.filtered_df[selected_columns]

    def highlight_row(row: Any) -> list[Any]:
        pred = row.get("Pred", None)
        gt = row.get("GT", None)

        if pred == "D" and gt == "D":
            return ["background-color: lightblue"] * len(row)
        elif pred == "ND" and gt == "ND":
            return ["background-color: lightgreen"] * len(row)
        elif pred == "ND" and gt == "D":
            return ["background-color: lightcoral"] * len(row)
        elif pred == "D" and gt == "ND":
            return ["background-color: lightyellow"] * len(row)
        else:
            return [""] * len(row)

    styled_df = listview_df.style.apply(highlight_row, axis=1)

    event = st.dataframe(
        styled_df,
        use_container_width=True,
        height=300,
        hide_index=False,
        on_select="rerun",
        selection_mode=["single-row"],
    )
    st.text("Blue: TP, Green: TN, Red: FN, Yellow: FP")

    # Check if a row is selected
    if event.selection and "rows" in event.selection:
        if event.selection["rows"]:
            previous_selected_row_index = st.session_state.get("selected_row_index", None)
            st.session_state.selected_row_index = event.selection["rows"][0]

            # Map the selected row index back to the original DataFrame index
            original_index = st.session_state.filtered_df.iloc[st.session_state.selected_row_index].name
            st.session_state.selected_row_index = original_index

            if previous_selected_row_index != st.session_state.selected_row_index:
                st.session_state.selection_source = "list"

    # Get the selected row based on the session state
    if st.session_state.selection_source == "list":
        selected_data = df.loc[st.session_state.selected_row_index]
        defect_number = selected_data["No"]
        if st.session_state.defect_number != str(defect_number):
            st.session_state.defect_number = str(defect_number)
            st.query_params.defect_number = str(defect_number)
            st.rerun()

    else:
        selected_data = df[df["No"] == df["No"].min()]

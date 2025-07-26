import os
import operator
import random
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st
from loguru import logger
from sklearn.cluster import DBSCAN

from ltt_ff_frontend.defect_review_ui import detail_view
from ltt_ff_frontend.helpers import api_helper


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)] + [160]


# Function to generate a list of colors for a given number of clusters
def generate_colors(num_clusters):
    random.seed(num_clusters)
    # Use the 'viridis' colormap which can handle a large number of distinct colors
    cmap = plt.get_cmap("hsv", num_clusters)
    # Generate the colors
    colors = [mpl.colors.to_hex(cmap(i)) for i in range(num_clusters)]
    # Shuffle the colors
    random.shuffle(colors)
    return colors


# define callback when threshold changes
def reload_data(df: pd.DataFrame):
    # Add "D/ND" column based on the threshold (Defect/Not defect)
    df["D/ND"] = df["Probability"] >= st.session_state.prob_threshold
    # Create the new column 'C/NC' based on the conditions provided (Correct/Not correct)
    df["C/NC"] = df[["Ans", "D/ND"]].apply(
        lambda x: "UNK" if x["Ans"] == "UNK" else (x["Ans"] == "T") == x["D/ND"], axis=1
    )

    # Convert back "D/ND" "C/NC" column to T/F
    df["D/ND"] = df["D/ND"].apply(lambda x: "UNK" if x == "UNK" else "T" if x else "F")
    df["C/NC"] = df["C/NC"].apply(lambda x: "UNK" if x == "UNK" else "T" if x else "F")

    # Select only the columns I want to display
    selected_columns = [
        "X",
        "Y",
        "UniqueID",
        "X_norm",
        "Y_norm",
        "ClassType",
        "Ans",
        "Probability",
        "D/ND",
        "C/NC",
        "No",
        "Cluster",
    ]
    st.session_state.filtered_df = df[selected_columns]


def app(result_dir: str, image_dir: str, selected_lot_id: str) -> None:
    
    defects = api_helper.get_lrf_data_lists(
        output_dir=result_dir,
        cols=["No", "UniqueID", "X", "Y", "ClassType"],
        include_prob=True,
        lot_id=selected_lot_id,
    )[0]
    db_metadata = api_helper.get_db_metadata_lists(output_dir=result_dir, lot_id=selected_lot_id)[0]

    # Extract relevant columns and convert "X" and "Y" to floats
    defect_data = [
        {
            "No": defect["No"],
            "UniqueID": defect["UniqueID"],
            "X": float(defect["X"]),
            "Y": float(defect["Y"]),
            "ClassType": defect["ClassType"],
            "Ans": defect["Ans"],
            "Probability": defect["Probability"],
        }
        for defect in defects
    ]

    # Convert to DataFrame
    df = pd.DataFrame(defect_data)

    # Set the "No" column as the index
    df.set_index("No", inplace=True)
    df["No"] = df.index

    # Ensure all columns have consistent data types
    df["No"] = df["No"].astype(int)
    df["UniqueID"] = df["UniqueID"].astype(str)
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["ClassType"] = df["ClassType"].astype(int)
    df["Ans"] = df["Ans"].astype(int)
    df["Probability"] = df["Probability"].astype(float)

    # Initialize session state for selected index
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = 0
    if "selected_map_index" not in st.session_state:
        st.session_state.selected_map_index = 0

    # Initialize session state for selection source
    if "selection_source" not in st.session_state:
        st.session_state.selection_source = ""
    # Initialize session state for selected folder
    if "result_dir" not in st.session_state:
        st.session_state.result_dir = ""
    # Initialize session state for selected lot
    if "lot_id" not in st.session_state:
        st.session_state.lot_id = ""
    # Initialize session state for probability threshold
    if "prob_threshold" not in st.session_state:
        st.session_state.prob_threshold = db_metadata.get("model_threshold", db_metadata.get("model_threshold_0", -1))
    # Ensure color_option is set in session state
    if "color_option" not in st.session_state:
        st.session_state.color_option = "ClassType"

    # Add "D/ND" column based on the threshold (Defect/Not defect)
    df["D/ND"] = df["Probability"] >= st.session_state.prob_threshold
    # Create the new column 'C/NC' based on the conditions provided (Correct/Not correct)
    df["C/NC"] = df[["Ans", "D/ND"]].apply(lambda x: -1 if x["Ans"] == -1 else x["Ans"] == x["D/ND"], axis=1)

    # Convert "Ans" "D/ND" "C/NC" column to T/F
    df["Ans"] = df["Ans"].apply(lambda x: "UNK" if x == -1 else "T" if x else "F")
    df["D/ND"] = df["D/ND"].apply(lambda x: "UNK" if x == -1 else "T" if x else "F")
    df["C/NC"] = df["C/NC"].apply(lambda x: "UNK" if x == -1 else "T" if x else "F")

    # Normalize coordinates
    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()
    df["X_norm"] = (df["X"] - x_min) / (x_max - x_min)
    df["Y_norm"] = (df["Y"] - y_min) / (y_max - y_min)

    # Use DBScan to identify clusters
    dbscan = DBSCAN(eps=50, min_samples=5)
    df["Cluster"] = dbscan.fit_predict(df[["X", "Y"]])

    # Count the total number of clusters
    total_clusters = df["Cluster"].nunique()

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

    threshold_col, filter_options_col, filter_operators_col, filter_value_col, filter_message_col, apply_col, remove_col = st.columns([1, 1, 1, 2, 1, 1, 1])

    # Add threshold selection
    with threshold_col:
        # TODO: figure out why this variable is not working
        _threshold = st.number_input(
            "Select Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.00001,
            format="%.5f",
            key="prob_threshold",
            on_change=reload_data(st.session_state.filtered_df),
        )

    # Add filter options
    with filter_options_col:
        # Define the columns I want to display
        specific_columns = ["UniqueID", "X", "Y", "ClassType", "Ans", "D/ND", "C/NC", "Cluster"]
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
        index = operator_columns.index(st.session_state.filter_operator) if st.session_state.filter_operator in operator_columns else 0
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
                if col_dtype.kind in "iuf": # numeric types
                    val = float(val)
                elif col_dtype.kind == "b": # boolean
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

    # Add message box showing active filters
    with filter_message_col:
        # TODO: To be fixed. Current method will cause message box to not appear if no values are filtered,
        #       even if filter is active. But this is unlikely to happen.
        if len(defect_data) != len(st.session_state.filtered_df):
            st.warning(f"Active filter: {st.session_state.filter_column} = {st.session_state.filter_value}")

    # List view
    st.subheader("List View")

    # Select only the columns I want to display
    selected_columns = ["No", "UniqueID", "X", "Y", "ClassType", "Ans", "Probability", "D/ND", "C/NC", "Cluster"]
    listview_df = st.session_state.filtered_df[selected_columns]

    event = st.dataframe(
        listview_df,
        use_container_width=True,
        height=300,
        hide_index=False,
        on_select="rerun",
        selection_mode=["single-row"],
    )

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
    defect_number = 0
    if st.session_state.selection_source == "list":
        selected_data = df.loc[st.session_state.selected_row_index]
        st.query_params.defect_no = st.session_state.selected_row_index
        defect_number = st.session_state.selected_row_index

    elif st.session_state.selection_source == "map":
        selected_data = df[df["No"] == st.session_state.selected_map_index]
        st.query_params.defect_no = st.session_state.selected_map_index
        defect_number = st.session_state.selected_map_index

    else:
        selected_data = df[df["No"] == df["No"].min()]

    # # Parse URL to get the 'lot' parameter
    # query_params = st.query_params
    # defect_number = query_params.get("defect_no", None)

    # # Find the index of the lot_name in filtered_folders
    # if defect_number:
    #     defect_number = int(defect_number)
    #     if defect_number not in df["No"].values:
    #         defect_number = df["No"].min()
    #     st.query_params.defect_no = defect_number
    #     selected_data = df[df["No"] == defect_number]

    # if selected_data is not None:
    #     detail_view.app(selected_data, image_dir, db_metadata["input_lrf_ext"])

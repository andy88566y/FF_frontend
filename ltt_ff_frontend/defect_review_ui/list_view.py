import random

import matplotlib
import pandas as pd
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from loguru import logger

from ltt_ff_frontend.read_defect import read_defects
from ltt_ff_frontend.defect_review_ui import detail_view
from ltt_ff_frontend.defect_ui.defect_ui_helper import get_probability


# Cache the read_defects function
@st.cache_data(ttl='300s')
def cached_read_defects(lrf_path):
    return read_defects(lrf_path)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [160]

# Function to generate a list of colors for a given number of clusters
def generate_colors(num_clusters):
    random.seed(num_clusters)
    # Use the 'viridis' colormap which can handle a large number of distinct colors
    cmap = plt.get_cmap('hsv', num_clusters)
    # Generate the colors
    colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(num_clusters)]
    # Shuffle the colors
    random.shuffle(colors)
    return colors

# define callback when threshold changes
def reload_data(df):
    threshold = st.session_state['prob_threshold']
    df["D/ND"] = df["Probability"] > threshold
    df["D/ND"] = df["D/ND"].apply(lambda x: 'T' if x else 'F')
    df['C/NC'] = ((df['D/ND'] == 'T') & (df['ClassType'] == 1)) | ((df['D/ND'] == 'F') & (df['ClassType'] != 1))
    df["C/NC"] = df["C/NC"].apply(lambda x: 'T' if x else 'F')

    # Select only the columns I want to display
    selected_columns = ["X", "Y", "X_norm", "Y_norm", "ClassType", "Cluster", "Probability", "D/ND", "C/NC", "No"]
    st.session_state.filtered_df = df[selected_columns]


def app(selected_folder, lrf_path):
    defects = cached_read_defects(lrf_path)

    # Extract relevant columns and convert "X" and "Y" to floats
    defect_data = [
        {"No": defect["No"], "X": float(defect["X"]), "Y": float(defect["Y"]), "ClassType": defect["ClassType"]}
        for defect in defects.values()
    ]

    # # Get defect IDs
    # defect_ids = [defect["No"] for defect in defect_data]

    # Get probabilities
    # probabilities = get_probability("/mnt/fs0/xxx", "xxx", "xxx", defect_ids)
    probabilities = [-1]*len(defect_data)

    # Add probabilities to defect_data
    for defect, probability in zip(defect_data, probabilities):
        defect["Probability"] = probability

    # Convert to DataFrame
    df = pd.DataFrame(defect_data)

    # Set the "No" column as the index
    df.set_index("No", inplace=True)
    df['No'] = df.index

    # Initialize session state for selected index
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = 0
    if "selected_map_index" not in st.session_state:
        st.session_state.selected_map_index = 0

    # Initialize session state for selection source
    if 'selection_source' not in st.session_state:
        st.session_state.selection_source = ''
    # Initialize session state for selected folder
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = ''
    # Initialize session state for probability threshold
    if 'prob_threshold' not in st.session_state:
        st.session_state.prob_threshold = 0.174
    # Ensure color_option is set in session state
    if "color_option" not in st.session_state:
        st.session_state.color_option = "Cluster"

    # Ensure all columns have consistent data types
    df["No"] = df["No"].astype(int)
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["ClassType"] = df["ClassType"].astype(int)
    df["Probability"] = df["Probability"].astype(float)

    # Add "D/ND" column based on the threshold (Defect/Not defect)
    df["D/ND"] = df["Probability"] > st.session_state.prob_threshold
    # Convert "D/ND" column to T/F
    df["D/ND"] = df["D/ND"].apply(lambda x: 'T' if x else 'F')
    # Create the new column 'C/NC' based on the conditions provided (Correct/Not correct)
    # TODO: need to switch to lrf classtype mapping instead of hardcoding
    df['C/NC'] = ((df['D/ND'] == 'T') & (df['ClassType'] == 1)) | ((df['D/ND'] == 'F') & (df['ClassType'] != 1))
    # Convert "C/NC" column to T/F
    df["C/NC"] = df["C/NC"].apply(lambda x: 'T' if x else 'F')

    # Normalize coordinates
    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()
    df["X_norm"] = (df["X"] - x_min) / (x_max - x_min)
    df["Y_norm"] = (df["Y"] - y_min) / (y_max - y_min)

    # Use DBScan to identify clusters
    dbscan = DBSCAN(eps=50, min_samples=5)
    df['Cluster'] = dbscan.fit_predict(df[['X', 'Y']])

    # Count the total number of clusters
    total_clusters = df['Cluster'].nunique()

    # Initialize session state for filter criteria
    if 'filter_column' not in st.session_state:
        st.session_state.filter_column = df.columns[0]
    if 'filter_value' not in st.session_state:
        st.session_state.filter_value = ''
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df

    # Reset session state values when changing folder
    previous_selected_folder = st.session_state.get("selected_folder", None)
    if previous_selected_folder != selected_folder:
        st.session_state.filter_column = df.columns[0]
        st.session_state.filter_value = ''
        st.session_state.filtered_df = df
        st.session_state.selection_source = ''
        st.session_state.selected_folder = selected_folder

    # Create three columsn (prob threshold, filter options, message to show filtered values)
    threshold_col, filter_options_col, filter_value_col, filter_message_col = st.columns([1, 1, 1, 1])

    # Add threshold selection
    with threshold_col:
        threshold = st.number_input(
            "Select Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.00001,
            format="%.5f",
            key='prob_threshold',
            on_change=reload_data(st.session_state.filtered_df)
        )

    # Add filter options
    with filter_options_col:
        # Define the columns I want to display
        specific_columns = ["X", "Y", "ClassType", "Cluster", "D/ND", "C/NC"]
        # Filter the DataFrame columns to only include the specific columns
        filtered_columns = [col for col in df.columns if col in specific_columns]

        # Use the filtered columns in the selectbox
        st.session_state.filter_column = st.selectbox(
            label="Filter options",
            options=filtered_columns,
            index=filtered_columns.index(st.session_state.filter_column),
        )

    # Add filter value text input, confirm button, and cancel button
    with filter_value_col:
        filter_value_input_col, confirm_col, cancel_col = st.columns([3, 1, 1])
        with filter_value_input_col:
            st.session_state.filter_value = st.text_input(
                label="Filter value",
                value=st.session_state.filter_value,
            )
        with confirm_col:
            if st.button(label="", icon=":material/check:", use_container_width=True):
                st.session_state.filtered_df = df[df[st.session_state.filter_column].astype(str) == st.session_state.filter_value]
                # st.session_state.filtered_df.set_index("No", inplace=True)
        with cancel_col:
            if st.button("", icon=":material/close:", use_container_width=True):
                st.session_state.filter_column = df.columns[0]
                st.session_state.filter_value = ''
                st.session_state.filtered_df = df
                st.rerun()

    # Add message box showing active filters
    with filter_message_col:
        # TODO: To be fixed. Current method will cause message box to not appear if no values are filtered,
        #       even if filter is active. But this is unlikely to happen.
        if len(defect_data) != len(st.session_state.filtered_df):
            st.warning(f'Active filter: {st.session_state.filter_column} = {st.session_state.filter_value}')

    # Create two columns (list view, map view)
    col1, col2 = st.columns([1, 1])

    # List view
    with col1:
        st.subheader("List View")

        # Select only the columns I want to display
        selected_columns = ["X", "Y", "ClassType", "Cluster", "Probability", "D/ND", "C/NC"]
        listview_df = st.session_state.filtered_df[selected_columns]

        event = st.dataframe(
            listview_df,
            use_container_width=True,
            height=300,
            hide_index=False,
            on_select="rerun",
            selection_mode=["single-row"]
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

    # Map view
    with col2:
        # Create two columns
        col21, col22, col23= st.columns([5,2,2])
        with col21:
            st.subheader("Map View")

        # Add a toggle button for ColorType/Cluster
        with col22:
            if st.button("ClassType", key="color_type_button", use_container_width=True):
                st.session_state.color_option = "ClassType"

        with col23:
            if st.button("Cluster", key="cluster_button", use_container_width=True):
                st.session_state.color_option = "Cluster"

        color_option = st.session_state.color_option
        # Define a color mapping for each ClassType and Cluster
        classType_mapping = {
            1: "#55ff7f",  # green
            2: "#0000ff",  # Blue
            "default": "#ffff00"  # Yellow
        }
        cluster_colors = generate_colors(total_clusters)

        # Apply the color mapping based on the selected option
        if color_option == "ClassType":
            st.session_state.filtered_df["color"] = st.session_state.filtered_df["ClassType"].map(lambda x: hex_to_rgb(classType_mapping.get(x, "#ffff00")))
        else:
            st.session_state.filtered_df["color"] = st.session_state.filtered_df["Cluster"].map(lambda x: hex_to_rgb("#808080") if x == -1 else hex_to_rgb(cluster_colors[x % len(cluster_colors)]))

        # Define the scatter plot layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            id = "defect-map",
            data=st.session_state.filtered_df,
            get_position=["X_norm", "Y_norm"],
            get_fill_color="color",
            pickable=True,
            radius_scale=10,
            radius_min_pixels=2,
            radius_max_pixels=10,
            auto_highlight=True,
        )

        # Define the deck.gl view
        view_state = pdk.ViewState(
            latitude=0.5,
            longitude=0.5,
            controller=True,
            zoom=7,
            pitch=0
        )

        # Render the deck.gl map without a base map
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_provider=None,
            tooltip={"text": "No: {No}\nClassType: {ClassType}\nX: {X}\nY: {Y}\nCluster: {Cluster}"},
        )
        event = st.pydeck_chart(r,height=300, on_select="rerun", selection_mode="single-object")

        # Check if the map is selected
        indices = event.selection.get("indices", {}).get("defect-map", [])
        if indices:
            previous_selected_map_index = st.session_state.get("selected_map_index", None)
            # Iterate over the objects to find the corresponding 'No' value
            for obj in event.selection.get("objects", {}).get("defect-map", []):
                st.session_state.selected_map_index = obj['No']
                break

            if previous_selected_map_index != st.session_state.selected_map_index:
                st.session_state.selection_source = "map"


    # Get the selected row based on the session state
    defect_number = 0
    if st.session_state.selection_source == "list":
        selected_data = df.loc[st.session_state.selected_row_index]
        st.query_params.defect_no = st.session_state.selected_row_index
        defect_number = st.session_state.selected_row_index

    elif st.session_state.selection_source == "map":
        selected_data = df[df['No'] == st.session_state.selected_map_index]
        st.query_params.defect_no = st.session_state.selected_map_index
        defect_number = st.session_state.selected_map_index

    else:
        selected_data = df[df['No'] == 1]

    # Parse URL to get the 'lot' parameter
    query_params = st.query_params
    defect_number = query_params.get('defect_no', None)

    # Find the index of the lot_name in filtered_folders
    if defect_number:
        defect_number = int(defect_number)
        if defect_number >= len(df):
            defect_number = 1
        st.query_params.defect_no = defect_number
        selected_data = df[df['No'] == defect_number]

    if selected_data is not None:
        detail_view.app(selected_data,selected_folder)

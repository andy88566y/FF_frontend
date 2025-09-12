import pandas as pd
import pydeck as pdk
import streamlit as st
from loguru import logger
from sklearn.cluster import DBSCAN

from ltt_ff_frontend.defect_review.list_view import generate_colors, hex_to_rgb


def app() -> None:
    if "color_option" not in st.session_state:
        st.session_state.color_option = "ClassType"
    col_header, col_classtype, col_cluster = st.columns([5, 2, 2])
    with col_header:
        st.subheader("Map View")

    # Add a toggle button for ColorType/Cluster
    with col_classtype:
        if st.button("ClassType", key="color_type_button", use_container_width=True):
            st.session_state.color_option = "ClassType"

    with col_cluster:
        if st.button("Cluster", key="cluster_button", use_container_width=True):
            st.session_state.color_option = "Cluster"

    color_option = st.session_state.color_option
    # Define a color mapping for each ClassType and Cluster
    classtype_mapping = {
        "D": "#ff0000",  # red
        "ND": "#55ff7f",  # green
        "UNK": "#808080",  # grey
    }

    # Convert to DataFrame
    if "defect_data" in st.session_state:
        df = pd.DataFrame(st.session_state.defect_data)
    else:
        return

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
    if st.session_state.result_dir and "prob_threshold" not in st.session_state:
        st.session_state.prob_threshold = st.session_state.db_metadata.get(
            "model_threshold", st.session_state.db_metadata.get("model_threshold_0", -1)
        )
    # Ensure color_option is set in session state
    if "color_option" not in st.session_state:
        st.session_state.color_option = "ClassType"

    # Use DBScan to identify clusters (might be slow need to add cache Test large dataset)
    dbscan = DBSCAN(eps=50, min_samples=5)
    df["Cluster"] = dbscan.fit_predict(df[["X", "Y"]])
    if st.session_state.model_count > 0:
        prob_df = pd.DataFrame(
            st.session_state.defect_prob["probability_list"][0],
            columns=[f"P_{model_name}" for model_name in st.session_state.models_name],
        )

        df = pd.concat([df, prob_df], axis=1)

    # Set the "No" column as the index
    df.set_index("No", inplace=True)
    df["No"] = df.index

    # Ensure all columns have consistent data types
    df["No"] = df["No"].astype(int)
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["ClassType"] = df["ClassType"].astype(int)
    df["GT"] = df["GT"].astype(int)
    if "UniqueID" in df.columns:
        df["UniqueID"] = df["UniqueID"].astype(str)
    if "Probability" in df.columns:
        df["Probability"] = df["Probability"].astype(float)
        df = df.rename(columns={"Probability": "P_rank"})

    # Normalize coordinates
    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()
    df["X_norm"] = (df["X"] - x_min) / (x_max - x_min)
    df["Y_norm"] = (df["Y"] - y_min) / (y_max - y_min)

    # Add "D/ND" column based on the threshold (Defect/Not defect)
    if st.session_state.result_dir:
        df["Pred"] = df["P_rank"] >= st.session_state.prob_threshold
        df["Pred"] = df["Pred"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")

    df["GT"] = df["GT"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")
    # Count the total number of clusters
    total_clusters = df["Cluster"].nunique()
    cluster_colors = generate_colors(total_clusters)
    if st.session_state.result_dir:
        selected_columns = [
            "No",
            "UniqueID",
            "X",
            "Y",
            "X_norm",
            "Y_norm",
            "ClassType",
            "GT",
            "Pred",
            "P_rank",
            "Cluster",
        ]
    else:
        selected_columns = ["No", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Cluster"]

    st.session_state.list_view_df = df
    st.session_state.filtered_df = df[selected_columns]
    # Apply the color mapping based on the selected option
    if color_option == "ClassType":
        st.session_state.filtered_df["color"] = st.session_state.filtered_df["GT"].map(
            lambda x: hex_to_rgb(classtype_mapping.get(x, "#808080"))
        )
    else:
        st.session_state.filtered_df["color"] = st.session_state.filtered_df["Cluster"].map(
            lambda x: hex_to_rgb("#808080") if x == -1 else hex_to_rgb(cluster_colors[x % len(cluster_colors)])
        )

    # Define the scatter plot layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        id="defect-map",
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
    view_state = pdk.ViewState(latitude=0.5, longitude=0.5, controller=True, zoom=7, pitch=0)

    # Render the deck.gl map without a base map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider=None,
        tooltip={"text": "No: {No}\nClassType: {ClassType}\nX: {X}\nY: {Y}\nCluster: {Cluster}"},
    )
    event = st.pydeck_chart(r, height=300, on_select="rerun", selection_mode="single-object")

    # Check if the map is selected
    indices = event.selection.get("indices", {}).get("defect-map", [])
    if indices:
        # Iterate over the objects to find the corresponding 'No' value
        previous_selected_map_index = st.session_state.get("defect_number", None)
        # Iterate over the objects to find the corresponding 'No' value
        for obj in event.selection.get("objects", {}).get("defect-map", []):
            st.session_state.selected_map_index = obj["No"]
            break

        if previous_selected_map_index != str(st.session_state.selected_map_index):
            st.session_state.selection_source = "map"
            defect_id = st.session_state.selected_map_index
            logger.info("map selected " + str(defect_id))
            st.session_state.defect_number = str(defect_id)
            st.query_params.defect_number = str(defect_id)
            st.rerun()

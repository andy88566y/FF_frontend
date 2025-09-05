import base64
import os
import re

import pandas as pd
import pydeck as pdk
import streamlit as st
from loguru import logger
from sklearn.cluster import DBSCAN

from ltt_ff_frontend.defect_review import list_view
from ltt_ff_frontend.defect_review.label_mapping import generate_lttswadc_html
from ltt_ff_frontend.defect_review.list_view import generate_colors, hex_to_rgb
from ltt_ff_frontend.defect_review.lrf_constant import lttswadc_map
from ltt_ff_frontend.defect_review_ui.defect_diff_viewer import draw_diff_img_plotly
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.title("Defect Review")

    col1, col2 = st.columns([1, 3])
    ###Mask info
    with col1:
        with st.container():
            query_params = st.query_params
            encoded_result_dir = query_params.get("result_dir", "")
            encoded_data_yaml = query_params.get("data_yaml", "")
            encoded_defect_id = query_params.get("defect_no", "")

            # Decode base64 parameters
            def decode_param(param: str) -> str:
                try:
                    return base64.urlsafe_b64decode(param.encode()).decode()
                except AttributeError:
                    return ""

            decoded_result_dir = decode_param(encoded_result_dir)
            decoded_data_yaml = decode_param(encoded_data_yaml)
            if encoded_defect_id:
                decoded_defect_id = int(encoded_defect_id)
            else:
                decoded_defect_id = None
            # Initialize session state
            if "result_dir" not in st.session_state:
                st.session_state.result_dir = decoded_result_dir
            if "data_yaml" not in st.session_state:
                st.session_state.data_yml = decoded_data_yaml

            # Input fields
            result_dir_input = st.text_input("Result Directory", value=st.session_state.result_dir)
            data_yaml_input = st.text_input("Data Yaml Path", value=st.session_state.data_yml)

            # Update query params if user changes input
            if result_dir_input != st.session_state.result_dir:
                st.session_state.result_dir = result_dir_input
                st.query_params["result_dir"] = base64.urlsafe_b64encode(result_dir_input.encode()).decode()
                st.rerun()

            if data_yaml_input != st.session_state.data_yml:
                st.session_state.data_yml = data_yaml_input
                st.query_params["data_yaml"] = base64.urlsafe_b64encode(data_yaml_input.encode()).decode()
                st.rerun()

            if not result_dir_input and not data_yaml_input:
                st.caption("Please input an Result Directory and Data yml to begin reviewing defects.")
                text_input_result_dir = ""
            else:
                text_input_result_dir = result_dir_input

            # TODO move out to another function and add cache
            dir_lots = []
            data_lots = []
            lot_lrf_path_map = {}
            if result_dir_input:
                dir_lots = [file.split(".")[0] for file in os.listdir(result_dir_input) if ".db" in file]
            if data_yaml_input:
                lot_lrf_path_map = api_helper.get_lot_lrf_paths(data_yaml_input)
                data_lots = list(lot_lrf_path_map.keys())

            if not dir_lots:
                lots = data_lots
            elif not data_lots:
                lots = dir_lots
            else:
                lots = list(set(dir_lots) & set(data_lots))

            if len(lots) >= 1:
                selected_lot_id = st.selectbox(label="Select a Lot ID", options=lots)
            else:
                selected_lot_id = None
                logger.warning("Cannot find any lots")
                return

            logger.info(f"Lot selected: {selected_lot_id}")
            model_num = 0
            r1_col1, r1_col2 = st.columns([2, 1])
            with r1_col1:
                if text_input_result_dir:
                    lrf_ext = api_helper.get_lot_lrf_ext(data_yaml_path=data_yaml_input, lot_id=selected_lot_id)
                    defects = api_helper.get_lrf_data_lists(
                        output_dir=text_input_result_dir,
                        cols=["No", "UniqueID", "X", "Y", "ClassType"],
                        include_prob=True,
                        lot_id=selected_lot_id,
                    )[0]
                    db_metadata = api_helper.get_db_metadata_lists(
                        output_dir=text_input_result_dir, lot_id=selected_lot_id
                    )[0]
                    defect_prob = api_helper.get_probabilities_per_model(
                        output_dir=text_input_result_dir, lot_id=selected_lot_id
                    )
                    model_num = len(defect_prob["probability_list"][0][0])
                    models_threshold = []
                    models_threshold_c = []
                    models_name = []
                    for x in range(model_num):
                        threshold_name = "model_threshold_" + str(x)
                        threshold_c_name = "model_threshold_c_" + str(x)
                        model_name = "model_name_" + str(x)
                        models_threshold.append(db_metadata[threshold_name])
                        models_threshold_c.append(db_metadata[threshold_c_name])
                        match = re.search(r"#([^#\.]+)\.", db_metadata[model_name])
                        if match:
                            models_name.append(match.group(1))
                        else:
                            models_name.append(db_metadata[model_name])
                    defect_data = [
                        {
                            "No": defect["No"],
                            "UniqueID": defect["UniqueID"],
                            "X": float(defect["X"]),
                            "Y": float(defect["Y"]),
                            "ClassType": defect["ClassType"],
                            "GT": defect["Ans"],
                            "Probability": defect["Probability"],
                        }
                        for defect in defects
                    ]
                else:
                    lrf_ext = api_helper.get_lot_lrf_ext(data_yaml_path=data_yaml_input, lot_id=selected_lot_id)
                    lrf_path = lot_lrf_path_map[selected_lot_id]
                    defects = api_helper.parse_lrf_data_lists(lrf_path=lrf_path)
                    if lrf_ext == "lrf":
                        defect_data = [
                            {
                                "No": defect["No"],
                                "X": float(defect["X"]),
                                "Y": float(defect["Y"]),
                                "ClassType": defect["ClassType"],
                                "GT": defect["isDefect"],  # valeu is 0 or 1
                            }
                            for defect in defects
                        ]
                    else:
                        defect_data = [
                            {
                                "No": defect["No"],
                                "UniqueID": defect["UniqueID"],
                                "X": float(defect["X"]),
                                "Y": float(defect["Y"]),
                                "ClassType": defect["ClassType"],
                                "GT": defect["isDefect"],  # valeu is 0 or 1
                            }
                            for defect in defects
                        ]

                if decoded_defect_id is not None:
                    st.session_state.defect_number = decoded_defect_id
                else:
                    st.session_state.defect_number = None

                no_list = [defect["No"] for defect in defect_data]
                default_index = (
                    no_list.index(st.session_state.defect_number) if st.session_state.defect_number in no_list else 0
                )
                defect_id = st.selectbox(label="Select a defect ID", options=no_list, index=default_index)
                # Update session state
                st.session_state.defect_number = defect_id

            with r1_col2:
                norm = st.toggle("Normalize", value=True)

        ###Label info
        st.title("LTTSWADC Map")
        styled_html = generate_lttswadc_html(lttswadc_map)
        st.markdown(styled_html, unsafe_allow_html=True)

        # MapView
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
        df = pd.DataFrame(defect_data)

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
        if text_input_result_dir and "prob_threshold" not in st.session_state:
            st.session_state.prob_threshold = db_metadata.get(
                "model_threshold", db_metadata.get("model_threshold_0", -1)
            )
        # Ensure color_option is set in session state
        if "color_option" not in st.session_state:
            st.session_state.color_option = "ClassType"

        # Use DBScan to identify clusters (might be slow need to add cache Test large dataset)
        dbscan = DBSCAN(eps=50, min_samples=5)
        df["Cluster"] = dbscan.fit_predict(df[["X", "Y"]])
        if model_num > 0:
            prob_df = pd.DataFrame(
                defect_prob["probability_list"][0], columns=[f"P_{models_name[i]}" for i in range(10)]
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
        if text_input_result_dir:
            df["Pred"] = df["P_rank"] >= st.session_state.prob_threshold
            df["Pred"] = df["Pred"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")

        df["GT"] = df["GT"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")
        # Count the total number of clusters
        total_clusters = df["Cluster"].nunique()
        cluster_colors = generate_colors(total_clusters)
        if text_input_result_dir:
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

            if previous_selected_map_index != st.session_state.selected_map_index:
                st.session_state.selection_source = "map"
                defect_id = st.session_state.selected_map_index
                logger.info("map selected" + str(defect_id))
                st.query_params.defect_no = defect_id
                st.session_state.defect_number = defect_id
                st.rerun()

    with col2:
        if defect_id != "":
            if lrf_ext == "lrf":
                draw_diff_img_plotly(data_yaml_input, selected_lot_id, defect_id, norm, diff_clip=0.3)
            else:
                row_index = df.index[df["No"] == int(defect_id)]
                unique_id = df.loc[row_index[0], "UniqueID"]
                draw_diff_img_plotly(data_yaml_input, selected_lot_id, unique_id, norm, diff_clip=0.3)

        ### Receipe area
        if text_input_result_dir and defect_id != "":
            row_number = df.index.get_loc(int(defect_id))
            selected_defect_prob = defect_prob["probability_list"][0][row_number]

            table_data = {
                "Model Name": models_name,
                "Defect Probability / Threshold": [
                    f"{selected_defect_prob[i]:.3f}/{models_threshold[i]}" for i in range(model_num)
                ],
                "Defect Probability / Threshold_C": [
                    f"{selected_defect_prob[i]:.3f}/{models_threshold_c[i]}" for i in range(model_num)
                ],
            }

            table_df = pd.DataFrame(table_data)

            # Apply styling and hide index
            styled_df = table_df.style.set_table_styles(
                [{"selector": "th, td", "props": [("text-align", "center")]}]
            ).hide(axis="index")

            # Convert to HTML
            html_table = styled_df.to_html()

            # Display using Streamlit
            st.title("Defect Probability vs Threshold Table")
            st.markdown(html_table, unsafe_allow_html=True)

        ### List view
        if text_input_result_dir:
            if not os.path.isdir(text_input_result_dir):
                raise ValueError(f"Input Result directory in text field is invalid: {text_input_result_dir}")

            logger.info("Input field params encoded and stored in URL.")
            list_view.app(text_input_result_dir, selected_lot_id, models_name, lrf_ext, df)
        else:
            list_view.app("", selected_lot_id, [], lrf_ext, df)

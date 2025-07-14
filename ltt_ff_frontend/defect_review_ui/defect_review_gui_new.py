import base64
import os
import re

import streamlit as st
from loguru import logger
import pandas as pd
import pydeck as pdk
from sklearn.cluster import DBSCAN


from ltt_ff_frontend.defect_review_ui import list_view_new
from ltt_ff_frontend.defect_review_ui.list_view_new import generate_colors, hex_to_rgb
from ltt_ff_frontend.defect_review_ui.lrf_constant import lttswadc_map
from ltt_ff_frontend.helpers import api_helper

def app() -> None:
    st.title("Defect Review new")
    
    col1, col2 = st.columns([1, 4])
    ###Mask info
    with col1:
        with st.container():
            if "result_dir" not in st.session_state:
                st.session_state.result_dir = ""
            if "image_dir" not in st.session_state:
                st.session_state.image_dir = ""
            encoded_result_dir_as_str = st.query_params.get("result_dir", None)
            encoded_image_dir_as_str = st.query_params.get("image_dir", None)

            decoded_result_dir_as_str, decoded_image_dir_as_str = "", ""
            if isinstance(encoded_result_dir_as_str, str):
                encoded_result_dir_as_bytes = str.encode(encoded_result_dir_as_str)
                decoded_result_dir_as_bytes = base64.urlsafe_b64decode(encoded_result_dir_as_bytes)
                decoded_result_dir_as_str = decoded_result_dir_as_bytes.decode()
            if isinstance(encoded_image_dir_as_str, str):
                encoded_image_dir_as_bytes = str.encode(encoded_image_dir_as_str)
                decoded_image_dir_as_bytes = base64.urlsafe_b64decode(encoded_image_dir_as_bytes)
                decoded_image_dir_as_str = decoded_image_dir_as_bytes.decode()

            text_input_result_dir = st.text_input(label="Result Directory", value=st.session_state.result_dir)
            if text_input_result_dir:
                text_input_result_dir = os.path.normpath(text_input_result_dir)
                st.query_params.result_dir = base64.urlsafe_b64encode(str.encode(text_input_result_dir)).decode()
            text_input_image_dir = st.text_input(label="Image Directory", value=st.session_state.image_dir)
            if text_input_image_dir:
                text_input_image_dir = os.path.normpath(text_input_image_dir)
                st.query_params.image_dir = base64.urlsafe_b64encode(str.encode(text_input_image_dir)).decode()
            if not text_input_result_dir or not text_input_image_dir:
                st.caption("Please input an Result Directory and Image Directory to begin reviewing defects.")
            
            lots = [file.split(".")[0] for file in os.listdir(text_input_result_dir) if ".db" in file]

            if len(lots) > 1:
                selected_lot_id = st.selectbox(label="Select a Lot ID", options=lots)
            else:
                selected_lot_id = lots[0]
            logger.info(f"Lot selected: {selected_lot_id}")

            # Ensure selected Lot ID matches Image Directory
            if re.search(re.escape(selected_lot_id), text_input_image_dir) is None:
                logger.error(f"Mismatch between Lot ID ({selected_lot_id}) and image directory ({text_input_image_dir}).")
                st.error(f"Mismatch between Lot ID ({selected_lot_id}) and image directory ({text_input_image_dir}).")
                return

        ###Label info
        with st.container():            
            indices = []
            labels = []

            for i in range(0, len(lttswadc_map), 8):
                index_row = [
                    f"<span style='color:red'>{j}</span>" if lttswadc_map[j][1] == 1 else str(j)
                    for j in range(i, min(i + 8, len(lttswadc_map)))
                ]
                label_row = [lttswadc_map[j][0] for j in range(i, min(i + 8, len(lttswadc_map)))]
                indices.append(index_row)
                labels.append(label_row)

            # Combine index and label rows into a single DataFrame
            table_rows = []
            for idx_row, lbl_row in zip(indices, labels):
                table_rows.append(idx_row)
                table_rows.append(lbl_row)

            df = pd.DataFrame(table_rows)

            # Display
            st.title("LTTSWADC Map (4x8 Table)")
            st.markdown(df.to_html(escape=False, index=False, header=False), unsafe_allow_html=True)

        if "color_option" not in st.session_state:
            st.session_state.color_option = "ClassType"
        with st.container():
            col21, col22, col23 = st.columns([5, 2, 2])
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
                "T": "#ff0000",  # red
                "F": "#55ff7f",  # green
                "UNK": "#808080",  # grey
            }

            defects = api_helper.get_lrf_data_lists(
                output_dir=text_input_result_dir,
                cols=["No", "UniqueID", "X", "Y", "ClassType"],
                include_prob=True,
                lot_id=selected_lot_id,
            )[0]
            db_metadata = api_helper.get_db_metadata_lists(output_dir=text_input_result_dir, lot_id=selected_lot_id)[0]

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
            cluster_colors = generate_colors(total_clusters)
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
            # Apply the color mapping based on the selected option
            if color_option == "ClassType":
                st.session_state.filtered_df["color"] = st.session_state.filtered_df["Ans"].map(
                    lambda x: hex_to_rgb(classType_mapping.get(x, "#808080"))
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
                previous_selected_map_index = st.session_state.get("selected_map_index", None)
                # Iterate over the objects to find the corresponding 'No' value
                for obj in event.selection.get("objects", {}).get("defect-map", []):
                    st.session_state.selected_map_index = obj["No"]
                    break

                if previous_selected_map_index != st.session_state.selected_map_index:
                    st.session_state.selection_source = "map"

    with col2:
        with st.container():
            st.write("This is the image area.")
        with st.container():
            st.write("This is the receipe area.")
        with st.container():
            if text_input_result_dir and text_input_image_dir:
                if not os.path.isdir(text_input_result_dir):
                    raise ValueError(f"Input Result directory in text field is invalid: {text_input_result_dir}")

                if not os.path.isdir(text_input_image_dir):
                    raise ValueError(f"Input Image directory in text field is invalid: {text_input_image_dir}")

                logger.info("Input field params encoded and stored in URL.")
                list_view_new.app(text_input_result_dir, text_input_image_dir, selected_lot_id)
            elif text_input_result_dir or text_input_image_dir:
                pass

            elif decoded_result_dir_as_str and decoded_image_dir_as_str:
                if not os.path.exists(decoded_result_dir_as_str):
                    raise ValueError(f"Result directory in URL is invalid: {decoded_result_dir_as_str}")

                if not os.path.isdir(decoded_image_dir_as_str):
                    raise ValueError(f"Image directory in URL is invalid: {decoded_image_dir_as_str}")

                logger.info("URL params successfully parsed.")
                st.session_state.result_dir = decoded_result_dir_as_str
                st.session_state.image_dir = decoded_image_dir_as_str
                st.rerun()

            else:
                pass
    

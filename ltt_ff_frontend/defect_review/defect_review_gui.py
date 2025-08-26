import base64
import os
import re

import streamlit as st
from loguru import logger
import pandas as pd
import pydeck as pdk
from sklearn.cluster import DBSCAN


from ltt_ff_frontend.defect_review import list_view
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
            def decode_param(param):
                try:
                    return base64.urlsafe_b64decode(param.encode()).decode()
                except Exception:
                    return ""

            decoded_result_dir = decode_param(encoded_result_dir)
            decoded_data_yaml = decode_param(encoded_data_yaml)
            decoded_defect_id = int(encoded_defect_id)
            # Initialize session state
            if "result_dir" not in st.session_state:
                st.session_state.result_dir = decoded_result_dir
            if "data_yaml" not in st.session_state:
                st.session_state.data_yml = decoded_data_yaml

            # Input fields
            result_dir_input = st.text_input("Result Directory", value=st.session_state.result_dir)
            data_yaml_input = st.text_input("Data Yaml Path",  value=st.session_state.data_yml)
            
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
            
            else:
                text_input_result_dir = result_dir_input

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

            if len(lots) > 1:
                selected_lot_id = st.selectbox(label="Select a Lot ID", options=lots)
            else:
                selected_lot_id = lots[0]
            logger.info(f"Lot selected: {selected_lot_id}")
            r1_col1, r1_col2 = st.columns([2, 1])
            with r1_col1:
                print(decoded_defect_id)
                if "defect_number" not in st.session_state or decoded_defect_id != None:
                    st.session_state.defect_number = decoded_defect_id            
                
                defect_id = st.text_input("Defect ID", value=st.session_state.defect_number)
                
            with r1_col2:
                norm = st.toggle("Normalize", value=True)
        ###Label info  
         
        indices = []     
        
        for i in range(0, len(lttswadc_map), 8):
            index_row = []
            for col_offset, j in enumerate(range(i, min(i + 8, len(lttswadc_map)))):
                edge_class = "left-edge" if col_offset == 0 else ""
                if lttswadc_map[j][1] == 1:
                    cell_html = f"""<div class='tooltip {edge_class}'><span style='color:red'>{j}</span><span class='tooltiptext'>{lttswadc_map[j][0]}</span></div>"""
                else:
                    cell_html = f"""<div class='tooltip {edge_class}'>{j}<span class='tooltiptext'>{lttswadc_map[j][0]}</span></div>"""
                index_row.append(cell_html)
            indices.append(index_row)


        # Create DataFrame
        df = pd.DataFrame(indices)

        # Display title
        st.title("LTTSWADC Map")

        # Display styled table with tooltips
        
        st.markdown(
            f"""
            <div style="max-width: 100%; overflow-x: auto;">
                <style>
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        table-layout: fixed;
                    }}
                    th, td {{
                        text-align: center;
                        padding: 6px;
                        border: 1px solid #ccc;
                        word-wrap: break-word;
                        font-size: 14px;
                        overflow: visible; /* Allow tooltips to overflow */
                    }}
                    .tooltip {{
                        position: relative;
                        display: inline-block;
                        cursor: pointer;
                        overflow: visible; /* Ensure tooltip is not clipped */
                    }}
                    .tooltip .tooltiptext {{
                        visibility: hidden;
                        background-color: #555;
                        color: #fff;
                        text-align: center;
                        border-radius: 6px;
                        padding: 5px;
                        position: absolute;
                        z-index: 1;
                        top: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        opacity: 0;
                        transition: opacity 0.3s;
                        white-space: nowrap;
                        margin-top: 6px;
                    }}
                    .tooltip.left-edge .tooltiptext {{
                        left: 0;
                        transform: none;
                    }}
                    .tooltip:hover .tooltiptext {{
                        visibility: visible;
                        opacity: 1;
                    }}
                </style>
                {df.to_html(escape=False, index=False, header=False)}
            </div>
            """,
            unsafe_allow_html=True
        )


        # MapView
        if "color_option" not in st.session_state:
            st.session_state.color_option = "ClassType"
        colHeader, colClassType, colCluster = st.columns([5, 2, 2])
        with colHeader:
            st.subheader("Map View")

        # Add a toggle button for ColorType/Cluster
        with colClassType:
            if st.button("ClassType", key="color_type_button", use_container_width=True):
                st.session_state.color_option = "ClassType"

        with colCluster:
            if st.button("Cluster", key="cluster_button", use_container_width=True):
                st.session_state.color_option = "Cluster"

        color_option = st.session_state.color_option
        # Define a color mapping for each ClassType and Cluster
        classType_mapping = {
            "D": "#ff0000",  # red
            "ND": "#55ff7f",  # green
            "UNK": "#808080",  # grey
        }
        if text_input_result_dir:       
            defects = api_helper.get_lrf_data_lists(
                output_dir=text_input_result_dir,
                cols=["No", "UniqueID", "X", "Y", "ClassType"],
                include_prob=True,
                lot_id=selected_lot_id,
            )[0]
            db_metadata = api_helper.get_db_metadata_lists(output_dir=text_input_result_dir, lot_id=selected_lot_id)[0]
            defect_prob = api_helper.get_probabilities_per_model(output_dir=text_input_result_dir, lot_id=selected_lot_id)
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
            df["UniqueID"] = df["UniqueID"].astype(str)
            df["Probability"] = df["Probability"].astype(float)

        else:
            lrf_path = lot_lrf_path_map[selected_lot_id]
            defects = api_helper.parse_lrf_data_lists(lrf_path=lrf_path)
            print(defects[0].keys())
            defect_data = [
                {
                    "No": defect["No"],
                    "X": float(defect["X"]),
                    "Y": float(defect["Y"]),
                    "ClassType": defect["ClassType"],
                    "GT": defect["isDefect"]
                }
                for defect in defects
            ]
            print(defect_data)

       
        # Extract relevant columns and convert "X" and "Y" to floats
        

        # Convert to DataFrame
        df = pd.DataFrame(defect_data)

        # Set the "No" column as the index
        df.set_index("No", inplace=True)
        df["No"] = df.index

        # Ensure all columns have consistent data types
        df["No"] = df["No"].astype(int)
        df["X"] = df["X"].astype(float)
        df["Y"] = df["Y"].astype(float)
        df["ClassType"] = df["ClassType"].astype(int)
        df["GT"] = df["GT"].astype(int)

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
            st.session_state.prob_threshold = db_metadata.get("model_threshold", db_metadata.get("model_threshold_0", -1))
        # Ensure color_option is set in session state
        if "color_option" not in st.session_state:
            st.session_state.color_option = "ClassType"

        # Add "D/ND" column based on the threshold (Defect/Not defect)
        if text_input_result_dir:
            df["Pred"] = df["P_rank"] >= st.session_state.prob_threshold
            df["Pred"] = df["Pred"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")
            
        df["GT"] = df["GT"].apply(lambda x: "UNK" if x == -1 else "D" if x else "ND")


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
        if text_input_result_dir:
            selected_columns = ["No", "UniqueID", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Pred", "P_rank", "Cluster"]
        else:
            selected_columns = ["No", "X", "Y", "X_norm", "Y_norm", "ClassType", "GT", "Cluster"]

        st.session_state.filtered_df = df[selected_columns]
        # Apply the color mapping based on the selected option
        if color_option == "ClassType":
            st.session_state.filtered_df["color"] = st.session_state.filtered_df["GT"].map(
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
            tooltip={"text": "No: {Nfo}\nClassType: {ClassType}\nX: {X}\nY: {Y}\nCluster: {Cluster}"},
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
            
            print(previous_selected_map_index)
            print(st.session_state.selected_map_index)

            if previous_selected_map_index != st.session_state.selected_map_index:
                st.session_state.selection_source = "map"
                defect_id = st.session_state.selected_map_index
                print("map selected" + str(defect_id))
                st.query_params.defect_no = defect_id
                st.rerun()

    with col2:     

        if defect_id != "":
            draw_diff_img_plotly(data_yaml_input, selected_lot_id, defect_id, norm, diff_clip=0.3)
        
        ### Receipe area
        if text_input_result_dir and defect_id != "":
            models_threshold = []
            models_threshold_c = []
            models_name = []
            selected_defect_prob = defect_prob["probability_list"][0][int(defect_id)-1]
            model_num = len(selected_defect_prob)
            for x in range(model_num):
                thresholdName = "model_threshold_" + str(x)
                thresholdcName = "model_threshold_c_" + str(x)
                modelName = "model_name_" + str(x)
                models_threshold.append(db_metadata[thresholdName])
                models_threshold_c.append(db_metadata[thresholdcName])
                #Todo, if match not found, just keep the model name
                models_name.append(re.search(r"#([^#\.]+)\.", db_metadata[modelName]).group(1))
            
            table_data = {
                "Model Name": models_name,
                "Defect Probability / Threshold": [f"{selected_defect_prob[i]:.3f}/{models_threshold[i]}" for i in range(model_num)],
                "Defect Probability / Threshold_C": [f"{selected_defect_prob[i]:.3f}/{models_threshold_c[i]}" for i in range(model_num)]
            }
            
            df = pd.DataFrame(table_data)

            # Apply styling and hide index
            styled_df = df.style.set_table_styles(
                [{'selector': 'th, td', 'props': [('text-align', 'center')]}]
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
            list_view.app(text_input_result_dir, None, selected_lot_id, models_name)
        else:
            list_view.app(None, data_yaml_input, selected_lot_id, None)

    

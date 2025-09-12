import base64
import os
import re

import streamlit as st
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    # Decode base64 parameters
    def decode_param(param: str) -> str:
        try:
            return base64.urlsafe_b64decode(param.encode()).decode()
        except AttributeError:
            return ""

    def get_defect_data(selected_lot_id: str, lot_lrf_path_map: dict) -> list[dict]:
        if st.session_state.result_dir:
            defects = api_helper.get_lrf_data_lists(
                output_dir=st.session_state.result_dir,
                cols=["No", "UniqueID", "X", "Y", "ClassType"],
                include_prob=True,
                lot_id=selected_lot_id,
            )[0]

            # Check if DB has same number of defects as data yaml's LRF
            lrf_defects = api_helper.parse_lrf_data_lists(lrf_path=lot_lrf_path_map[selected_lot_id])
            if len(defects) != len(lrf_defects):
                raise ValueError(
                    f"Unequal number of defects in DB ({len(defects)}) vs LRF ({len(lrf_defects)}). ",
                    "Please re-run inference using new LRF.",
                )

            st.session_state.db_metadata = api_helper.get_db_metadata_lists(
                output_dir=st.session_state.result_dir, lot_id=selected_lot_id
            )[0]
            st.session_state.defect_prob = api_helper.get_probabilities_per_model(
                output_dir=result_dir_input, lot_id=selected_lot_id
            )
            st.session_state.model_count = len(st.session_state.defect_prob["probability_list"][0][0])
            st.session_state.models_threshold = []
            st.session_state.models_threshold_c = []
            st.session_state.models_name = []
            st.session_state.selected_lot_lrf_ext = st.session_state.db_metadata["input_lrf_ext"]
            for x in range(st.session_state.model_count):
                st.session_state.models_threshold.append(st.session_state.db_metadata[f"model_threshold_{x}"])
                st.session_state.models_threshold_c.append(st.session_state.db_metadata[f"model_threshold_c_{x}"])
                match = re.search(r"#([^#\.]+)\.", st.session_state.db_metadata[f"model_name_{x}"])
                if match:
                    st.session_state.models_name.append(match.group(1))
                else:
                    st.session_state.models_name.append(st.session_state.db_metadata[f"model_name_{x}"])
            return [
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
            lrf_ext = api_helper.get_lot_lrf_ext(data_yaml_path=st.session_state.data_yaml, lot_id=selected_lot_id)
            lrf_path = lot_lrf_path_map[selected_lot_id]
            defects = api_helper.parse_lrf_data_lists(lrf_path=lrf_path)
            st.session_state.model_count = 0
            st.session_state.selected_lot_lrf_ext = lrf_ext
            if lrf_ext == "lrf":
                return [
                    {
                        "No": defect["No"],
                        "X": float(defect["X"]),
                        "Y": float(defect["Y"]),
                        "ClassType": defect["ClassType"],
                        "GT": defect["isDefect"],  # value is 0 or 1
                    }
                    for defect in defects
                ]
            else:
                return [
                    {
                        "No": defect["No"],
                        "UniqueID": defect["UniqueID"],
                        "X": float(defect["X"]),
                        "Y": float(defect["Y"]),
                        "ClassType": defect["ClassType"],
                        "GT": defect["isDefect"],  # value is 0 or 1
                    }
                    for defect in defects
                ]

    query_params = st.query_params
    url_result_dir = decode_param(query_params.get("result_dir", ""))
    url_data_yaml = decode_param(query_params.get("data_yaml", ""))
    url_defect_number = query_params.get("defect_number", None)

    # Initialize session state
    if "result_dir" not in st.session_state:
        st.session_state.result_dir = url_result_dir
    if "data_yaml" not in st.session_state:
        st.session_state.data_yaml = url_data_yaml
    if "defect_number" not in st.session_state:
        st.session_state.defect_number = url_defect_number

    # Input fields
    result_dir_input = st.text_input("Result Directory", value=st.session_state.result_dir)
    data_yaml_input = st.text_input("Data Yaml Path", value=st.session_state.data_yaml)

    # Update query params if user changes input
    if result_dir_input != st.session_state.result_dir:
        st.session_state.result_dir = result_dir_input
        st.query_params["result_dir"] = base64.urlsafe_b64encode(result_dir_input.encode()).decode()
        st.rerun()

    if data_yaml_input != st.session_state.data_yaml:
        st.session_state.data_yaml = data_yaml_input
        st.query_params["data_yaml"] = base64.urlsafe_b64encode(data_yaml_input.encode()).decode()
        st.rerun()

    if not result_dir_input and not data_yaml_input:
        st.caption("Please input an Result Directory and Data yaml to begin reviewing defects.")
        return
    elif result_dir_input and not data_yaml_input:
        st.error("Data yaml path is required.")
        return

    # TODO move out to another function and add cache
    lot_lrf_path_map = api_helper.get_lot_lrf_paths(data_yaml_input)
    lots = list(lot_lrf_path_map.keys())

    if result_dir_input:
        result_dir_lots = [lot.split(".")[0] for lot in os.listdir(result_dir_input) if ".db" in lot]
        lots = list(set(lots) & set(result_dir_lots))

    if len(lots) >= 1:
        selected_lot_id = st.selectbox(label="Select a Lot ID", options=lots)
        st.session_state.selected_lot_id = selected_lot_id
    else:
        raise ValueError("No lots were found or no intersecting lots in Result Directory and Data Yaml.")
    logger.info(f"Lot selected: {selected_lot_id}")

    st.session_state.defect_data = get_defect_data(selected_lot_id=selected_lot_id, lot_lrf_path_map=lot_lrf_path_map)

    lot_id_options_col, norm_toggle_col = st.columns([2, 1])
    with lot_id_options_col:
        def_no_list = [str(defect["No"]) for defect in st.session_state.defect_data]
        default_index = (
            def_no_list.index(st.session_state.defect_number) if st.session_state.defect_number in def_no_list else 0
        )
        defect_number = st.selectbox(label="Select a defect no.", options=def_no_list, index=default_index)
        if str(defect_number) != st.session_state.defect_number:
            st.session_state.defect_number = str(defect_number)
            st.query_params["defect_number"] = str(defect_number)
            st.rerun()

    with norm_toggle_col:
        st.session_state.norm = st.toggle("Normalize", value=True)

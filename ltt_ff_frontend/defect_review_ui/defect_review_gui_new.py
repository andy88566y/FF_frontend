import base64
import os
import re

import streamlit as st
from loguru import logger
import pandas as pd

from ltt_ff_frontend.defect_review_ui import list_view_new
from ltt_ff_frontend.defect_review_ui.lrf_constant import lttswadc_map

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

        with st.container():
            st.write("This is the defect location")

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
    

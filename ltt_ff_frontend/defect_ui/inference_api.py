import streamlit as st
from loguru import logger

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper





def app() -> None:
    logger.debug("Opening Inference API page")

    st.title("BDDetector False Filtering API")
    st.caption("API for running inference and fine-tuning False Filtering models.")

    st.header("Inference")


    with st.expander("Run inference on images"):

        st.markdown(":violet[Runs inference on images in the specified directory.]")

        st.header("Parameters")

        image_dir = st.text_input('Image directory', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532', help='The directory that contains the Images folder.')
        lrf_path = st.text_input('.lrf path', value='/mnt/fs0/x9u_detection_result/N3_M2-6_20240923_000532/N3_M2-6_20240923_000532_classified.lrf', help='Absolute path to the selected .lrf file.')
        lot_id = st.text_input('Lot ID', value='N3_M2-6_20240923_000532', help='Name of the lot of images to run inference on.')
        output_dir = st.text_input('Output directory', value='/mnt/fs0/shawn', help='The directory to store generated .lrf and .db')
        inference_batch_size = st.select_slider("Inference batch size", [4, 8, 16, 32])

        # selected_lot = st.selectbox(
        #     "Which lot do you want to run inference on?",
        #     helper.get_all_lots()
        # )
        # helper.gap(2)

        slider_threshold = st.slider("Select confidence threshold:", 0.0, 1.0, 0.5)
        st.caption(f"Probabilities above :blue[{slider_threshold}] will be considered defects.")
        helper.gap(2)

        # # add model selection here in the future
        # abs_model_paths = ...
        # model_basename: list[str] = []
        # selected_model = st.selectbox(
        #     "[Placeholder] Which model do you want to run?",
        #     model_basename
        # )
        # helper.gap(2)

        overwrite = st.toggle(label="Overwrite files in output directory", value=True)
        helper.gap(2)

        if st.button("Run inference", type='primary'):
            request = helper.request_inference(image_dir,
                                           lrf_path,
                                           lot_id,
                                           output_dir,
                                           inference_batch_size,
                                           slider_threshold,
                                           overwrite)
            if request.json().get('status') == 'error':
                code = request.json().get('code')
                message = request.json().get('message')
                st.text(f'Error code: {code}\nError message: {message}')
            else:
                inference_id = request.json().get('inference_id')
                st.text(f'Inference ID: {inference_id}')

            # st.divider()
            # # TODO: check with Carl on how to get inference progress
            # st.warning("TODO: Update progress bar")
            # progress_text = 'Inference in progress...'
            # progress_bar = st.progress(0.0, text=progress_text)

    with st.expander("Current jobs"):
        infer_id = st.text_input("Inference ID", help="The unique number generated after click Generate .lrf", value = '')
        if st.button("Check status"):
            status = helper.request_inference_status(infer_id)
            st.json(status.json())

    if st.button('Check all inference jobs'):
        all_statuses = helper.request_all_inference_statuses()
        detail_status_df = pd.DataFrame()
        brief = ['inference_id', 'status', 'processed_images', 'total_images']

        reorder = ['start_time_ymd','inference_id', 'status', 'processed_images', 'total_images', 'progress',
        'estimated_time_remaining', 'lot_id', 'output_dir', 'error_message' ]

        for status in all_statuses:
            new_row = pd.DataFrame([status])
            detail_status_df = pd.concat([detail_status_df, new_row], ignore_index=True)

        detail_status_df['start_time'] = pd.to_datetime(detail_status_df['start_time'])
        detail_status_df['start_time_ymd'] = detail_status_df['start_time'].dt.time
        detail_status_df = detail_status_df.iloc[::-1].reset_index(drop=True)
        status_df = detail_status_df[brief]

        st.session_state.status_df = status_df
        st.session_state.detail = detail_status_df[reorder]

    if "status_df" not in st.session_state:
        st.session_state.status_df = pd.DataFrame(columns=['inference_id', 'status', 'processed_images', 'total_images'])
    if "detail" not in st.session_state:
        st.session_state.detail = pd.DataFrame(columns = ['start_time_ymd','inference_id', 'status', 'processed_images', 'total_images', 'progress',
        'estimated_time_remaining', 'lot_id', 'output_dir', 'error_message'])


    event = st.dataframe(
        st.session_state.status_df,
        key = "statuses",
        on_select = "rerun",
        selection_mode = "multi-row",
    )
    if event and event.selection:
        # Check if the "row" value's list is not empty
        if event.selection["rows"]:
            # Extract the selected rows based on the indices
            selected_rows = st.session_state.detail.iloc[event.selection["rows"]]
            transposed_detail = selected_rows.T
            st.write(transposed_detail)
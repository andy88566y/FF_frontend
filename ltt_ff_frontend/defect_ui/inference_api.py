from typing import Any

import pandas as pd
import streamlit as st
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper


def create_job_list(paged_statuses: dict[str, Any], brief: list[str], inf_keys: list[str]):
    # Retrieve the whitelisted dictionaries
    if paged_statuses:
        inf_keys = list(next(iter(paged_statuses.values())).keys())
        inf_keys.append('inference_id')
        if 'progress' in inf_keys:
            inf_keys.remove('progress')

    whitelist_status_df = pd.DataFrame(columns = inf_keys)

    for inference_id, status in paged_statuses.items():
        status['inference_id'] = inference_id
        new_row = pd.DataFrame([status])
        if not new_row.empty and not new_row.isna().all().all():
            whitelist_status_df = pd.concat([whitelist_status_df, new_row], ignore_index=True)

    # convert start_time float to date time
    for column in whitelist_status_df.columns:
        if whitelist_status_df[column].dtype == 'object':
            whitelist_status_df[column] = whitelist_status_df[column].astype(str)

    if not whitelist_status_df.empty:
        whitelist_status_df['start_time'] = pd.to_datetime(whitelist_status_df['start_time'], unit='s')
        whitelist_status_df['start_time'] = whitelist_status_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        whitelist_status_df['progress_bar'] = whitelist_status_df['status'].apply(lambda x: helper.return_status_style(x))
        whitelist_status_df = whitelist_status_df.sort_values(by='start_time', ascending=False).reset_index(drop=True)

        status_df = whitelist_status_df[brief]

        st.session_state.status_df_inf = status_df
        st.session_state.whitelist_inf = whitelist_status_df[inf_keys]

def app() -> None:
    logger.debug("Loading Inference Dashboard...")
    st.title("False Filter Inference")
    st.caption("Inference lot data with selected model")

    r1_col1, r1_col2, r1_col3 = st.columns([3, 2, 2])
    with r1_col1:
        inf_base_model = st.selectbox("Base model", options=helper.get_base_models(), index=0,
                                        format_func=lambda x: x.replace("#", " "))
    with r1_col2:
        inf_filter_threshold = st.number_input("Confidence threshold:", 0.0, 1.0, 0.05, 0.00001, format="%.5f", help="Probabilities above threshold will be considered as defects.")
    with r1_col3:
        inf_overwrite = st.toggle(label="Overwrite files in output directory", value=False)

    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        inf_lot_id = st.text_input(label='Lot ID', value='', help='Name of the lot of defect images.')
    with r2_col2:
        inf_output_dir = st.text_input(label='Result directory', value='/mnt/dbpc/xxx', help='The directory to store generated .lrf and .db files.')

    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        inf_image_dir = st.text_input(label='Image directory', value='/mnt/dbpc/xxx', help='The directory that contains the Images folder.')
    with r3_col2:
        inf_lrf_path = st.text_input(label='.lrf path', value='/mnt/dbpc/xxx', help='Absolute path to the selected .lrf file.')

    if st.button("Start Inference Job", type="primary"):
        request = helper.request_inference(base_model=inf_base_model,
                                           lot_id=inf_lot_id,
                                           output_dir=inf_output_dir,
                                           image_dir=inf_image_dir,
                                           lrf_path=inf_lrf_path,
                                           confidence_threshold=inf_filter_threshold,
                                           overwrite=inf_overwrite)

        if request.json().get('status') == 'error':
            code = request.json().get('code')
            message = request.json().get('message')
            st.text(f'Error code: {code}\nError message: {message}')
        else:
            inference_id = request.json().get('inference_id')
            st.text(f'Inference Job ID: {inference_id}')

    st.divider()

    col1, col2 = st.columns(2, vertical_alignment='bottom')
    brief = ['inference_id', 'status','progress_bar', 'processed_images', 'total_images']
    inf_keys = ['inference_id']
    with col1:
        if st.button('Check all inference jobs'):
            page_size = 10
            current_page = 1
            paged_statuses = helper.request_paginated_inference_status(page_size, current_page)
            create_job_list(paged_statuses, brief, inf_keys)

    progress_column = st.column_config.ProgressColumn(
        label='progress_bar',
        min_value=0,
        max_value=100
    )

    if 'status_df_inf' not in st.session_state:
        st.session_state.status_df_inf = pd.DataFrame(columns=brief)
    if 'whitelist_inf' not in st.session_state:
        st.session_state.whitelist_inf = pd.DataFrame(columns=inf_keys)

    # Pagination settings
    with col2:
        if not st.session_state.status_df_inf.empty:
            page_size = 10
            current_page = st.number_input('Page number', min_value=1, value=1, step=1)
            paged_statuses = helper.request_paginated_inference_status(page_size, current_page)
            create_job_list(paged_statuses)

    st.header('All inference jobs') if not st.session_state.status_df_inf.empty else st.write('')

    # Selection to find more detail
    event_inf = st.dataframe(
        st.session_state.status_df_inf,
        key = 'statuses_inference',
        on_select = 'rerun',
        selection_mode = 'multi-row',
        use_container_width=True,
        column_config={'progress_bar': progress_column}
    ) if not st.session_state.status_df_inf.empty else st.write('')

    if event_inf and event_inf.selection:
    # Check if the 'row' value's list is not empty
        if event_inf.selection['rows']:
            # Extract the selected rows based on the indices
            selected_indices = [st.session_state.status_df_inf.index[i] for i in event_inf.selection['rows']]
            selected_rows = st.session_state.whitelist_inf.loc[selected_indices]
            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width=True)
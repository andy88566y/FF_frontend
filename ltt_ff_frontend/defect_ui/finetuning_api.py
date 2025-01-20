from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper

def create_job_list(paged_statuses: dict[str, Any], brief: list[str], fin_keys: list[str]):

    # Retrieve the whitelisted dictionaries
    if paged_statuses:
        fin_keys = list(next(iter(paged_statuses.values())).keys())
        fin_keys.append('training_id')
        if 'progress'in fin_keys:
            fin_keys.remove('progress')
        if 'estimated_time_remaining' in fin_keys:
            fin_keys.remove('estimated_time_remaining')

    whitelist_status_df = pd.DataFrame(columns = fin_keys)

    for training_id, status in paged_statuses.items():
        status['training_id'] = training_id
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

        st.session_state.status_df_fin = status_df
        st.session_state.whitelist_fin = whitelist_status_df[fin_keys]


def app() -> None:
    logger.debug("Loading Fine-Tuning Dashboard...")
    st.title("False Filter Fine-Tuning")
    st.caption("Train new model with selected lot data")

    r1_col1, r1_col2 = st.columns([3, 2])

    with r1_col1:
        ft_base_model = st.selectbox("Base model", options=helper.get_base_models(), index=0,
                                     format_func=lambda x: x.replace("#", " "))

    yaml_help_text = '''
    **Example of a valid .yaml config file:**\n
    data_paths:\n
    \- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000\n
    \- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000
'''

    with r1_col2:
        # TODO: Add yaml format help
        ft_configfile = st.file_uploader("Upload Multi-lot Fine-Tuning Config (.yaml)", type=".yaml", help=yaml_help_text)

    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns([1, 1, 2, 2])
    with r2_col1:
        ft_site = st.text_input("Site", max_chars=20)
    with r2_col2:
        ft_tool = st.text_input("Tool", value ="x9u", max_chars=10)
    with r2_col3:
        ft_techlayer = st.text_input("Tech Layer", max_chars=50)
    with r2_col4:
        ft_layergroup = st.text_input("Layer Group", max_chars=50)

    with st.expander("Fine-Tuning Parameters"):
        ft_epochs = st.number_input('Epochs', value=10)
        ft_lr = st.number_input('Learning Rate', value=0.0001, step=0.0001, format="%0.4f")

    if ft_configfile is not None:
        ft_config = yaml.load(ft_configfile, Loader=yaml.Loader)
        st.json(ft_config)

    if st.button("Start Fine-Tuning Job", type="primary"):
        request = helper.request_finetune(base_model=ft_base_model,
                                          model_naming=(ft_site, ft_tool, ft_techlayer, ft_layergroup),
                                          multilot_config=ft_config,
                                          epochs=ft_epochs, lr=ft_lr)

        if request.json().get('status') == 'error':
            code = request.json().get('code')
            message = request.json().get('message')
            st.text(f'Error code: {code}\nError message: {message}')
        else:
            training_id = request.json().get('training_id')
            st.text(f'Training Job ID: {training_id}')

    st.divider()

    # TODO: Simplify below

    col1, col2 = st.columns(2, vertical_alignment='bottom')

    brief = ['training_id', 'status','progress_bar', 'processed_images', 'total_images']
    fin_keys = ['training_id']
    with col1:
        if st.button('Check all finetuning jobs'):
            page_size = 10
            current_page = 1
            paged_statuses = helper.request_paginated_finetuning_status(page_size, current_page)
            create_job_list(paged_statuses,brief,fin_keys)

    progress_column = st.column_config.ProgressColumn(
        label='progress_bar',
        min_value=0,
        max_value=100
    )

    if 'status_df_fin' not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame(columns = brief)
    if 'whitelist_fin' not in st.session_state:
        st.session_state.whitelist_fin = pd.DataFrame(columns = fin_keys)

    # Pagination settings
    with col2:
        if not st.session_state.status_df_fin.empty:
            page_size = 10
            current_page = st.number_input('Page number', min_value=1, value=1, step=1)
            paged_statuses = helper.request_paginated_finetuning_status(page_size, current_page)
            create_job_list(paged_statuses)


    st.header('All fine-tuning jobs')  if not st.session_state.status_df_fin.empty else st.write('')

    # Selection to find more detail
    event_fin = st.dataframe(
        st.session_state.status_df_fin,
        key = 'statuses_finetuning',
        on_select = 'rerun',
        selection_mode = 'multi-row',
        use_container_width=True,
        column_config={'progress_bar': progress_column}
    ) if not st.session_state.status_df_fin.empty else st.write('')

    if event_fin and event_fin.selection:
        if event_fin.selection['rows']:
            selected_indices = [st.session_state.status_df_fin.index[i] for i in event_fin.selection['rows']]
            selected_rows = st.session_state.whitelist_fin.loc[selected_indices]
            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width= True )

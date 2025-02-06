from typing import Any
from pprint import pformat

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.defect_ui import defect_ui_helper as helper

def create_job_list(paged_statuses: dict[str, Any], brief: list[str]) -> None:

    detailed_status_df = pd.DataFrame.from_dict(paged_statuses).T

    if not detailed_status_df.empty:

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        detailed_status_df['start_time'] = pd.to_datetime(detailed_status_df['start_time'], unit='s').dt.floor('s')
        detailed_status_df['start_time'] = detailed_status_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

        # Get current and total epoch to caluclate progress
        current_epoch = detailed_status_df['current_epoch']
        total_epochs = detailed_status_df['total_epochs']
        detailed_status_df['progress_bar'] = detailed_status_df['status'].apply(lambda x: helper.return_finetune_status_style(x, current_epoch, total_epochs))

        # Convert model name to user-readable format
        detailed_status_df['base_model_name'] = detailed_status_df['base_model_name'].apply(helper.format_model_name)
        detailed_status_df['output_model_name'] = detailed_status_df['output_model_name'].apply(helper.format_model_name)

        # Add model name details to detailed status table
        detailed_status_df['site'] = detailed_status_df['base_model_name'].apply(lambda x: helper.filter_details(x, 'site'))
        detailed_status_df['tool'] = detailed_status_df['base_model_name'].apply(lambda x: helper.filter_details(x, 'tool'))
        detailed_status_df['tech_layer'] = detailed_status_df['base_model_name'].apply(lambda x: helper.filter_details(x, 'tech_layer'))
        detailed_status_df['layer_group'] = detailed_status_df['base_model_name'].apply(lambda x: helper.filter_details(x, 'layer_group'))

        # Sort jobs by start time
        detailed_status_df = detailed_status_df.sort_values(by='start_time', ascending=False).reset_index(drop=False)

        # Rename index column so that detailed status table will show 'training_id' instead of 'index'
        detailed_status_df = detailed_status_df.rename(columns={'index': 'training_id'})

        # Format training info (from yaml config) to be easily readable
        detailed_status_df['training_info'] = detailed_status_df['training_info'].map(lambda x: pformat(x))

        # Convert start time from seconds to human-readable format and change timezone to UTC+8
        if 'end_time' in detailed_status_df.columns:
            detailed_status_df['end_time'] = pd.to_datetime(detailed_status_df['end_time'], unit='s').dt.floor('s')
            detailed_status_df['end_time'] = detailed_status_df['end_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

        # Rename epoch loss & validation loss column to 'debug' as it is for internal use only
        if 'training_history' in detailed_status_df.columns:
            detailed_status_df = detailed_status_df.rename(columns={'training_history': 'debug'})
            detailed_status_df['debug'] = detailed_status_df['debug'].map(lambda x: pformat(x))

        # Update brief job list
        st.session_state.status_df_fin = detailed_status_df[brief]

        # Don't show progress bar in the detailed status table
        detailed_status_df = detailed_status_df.drop(columns=['progress_bar'])

        # Sort the items to show based on what may be important to user
        st.session_state.detailed_df_fin = detailed_status_df.reindex(columns=['training_id',
                                                                               'status',
                                                                               'start_time',
                                                                               'end_time',
                                                                               'current_epoch',
                                                                               'total_epochs',
                                                                               'base_model_name',
                                                                               'output_model_name',
                                                                               'site',
                                                                               'tool',
                                                                               'tech_layer',
                                                                               'layer_group',
                                                                               'batch_size',
                                                                               'learning_rate',
                                                                               'training_info',
                                                                               'debug',])

def app() -> None:
    logger.debug("Loading Fine-Tuning Dashboard...")
    st.title("False Filter Fine-Tuning")
    st.caption("Train new model with selected lot data")

    r1_col1, r1_col2 = st.columns([3, 2])

    with r1_col1:
        ft_base_model = st.selectbox("Base model", options=helper.get_base_models(), index=0,
                                     format_func=helper.format_model_name)

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
        if helper.check_valid_lrf_in_yaml(ft_config):
            logger.info("All lrf_paths in .yaml config file are valid!")
        st.json(ft_config)

    if st.button("Start Fine-Tuning Job", type="primary"):

        # Validate user input first
        required_input = [ft_site, ft_tool, ft_techlayer, ft_layergroup, ft_configfile]
        for item in required_input:
            if not item:
                logger.error('Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file.')
                st.error('Missing user input detected. Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file.')
                return

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

    col1, col2 = st.columns(2, vertical_alignment='bottom')

    # headers required for the brief job descriptions
    brief = ['training_id', 'status','progress_bar']

    with col1:
        if st.button('Check all finetuning jobs'):
            page_size = 10
            current_page = 1
            paged_statuses = helper.request_paginated_finetuning_status(page_size, current_page)
            create_job_list(paged_statuses, brief)

    progress_column = st.column_config.ProgressColumn(
        label='progress_bar',
        min_value=0,
        max_value=100
    )

    if 'status_df_fin' not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame()
    if 'detailed_df_fin' not in st.session_state:
        st.session_state.detailed_df_fin = pd.DataFrame()

    # Pagination settings
    with col2:
        if not st.session_state.status_df_fin.empty:
            page_size = 10

            current_page = st.number_input('Page number', min_value=1, value=1, step=1)
            paged_statuses = helper.request_paginated_finetuning_status(page_size, current_page)
            create_job_list(paged_statuses, brief)

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
            selected_rows = st.session_state.detailed_df_fin.loc[selected_indices]
            transposed_detail = selected_rows.T
            st.dataframe(transposed_detail, use_container_width= True )

import typing
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ltt_ff_frontend.constant import ST_DATAFRAME_ROW_HEIGHT, TrainingOption
from ltt_ff_frontend.datamodel.ff_core.request import FFCoreTrainingRequest
from ltt_ff_frontend.helpers import api_helper, training_helper
from ltt_ff_frontend.shared_components import helper, stop_job_button


if typing.TYPE_CHECKING:
    from ltt_ff_frontend.constant import NameNumInputParamsConfig


def app() -> None:
    logger.debug("Loading Training Dashboard...")
    st.title("False Filter Training By Config")
    st.caption("Train new model with selected lot data")

    training_option_val = st.segmented_control(
        label="Training option", options=[option.value for option in TrainingOption], default="Fine-tune"
    )

    if training_option_val is None:
        st.error("Please select a training method!")
        return

    st.divider()

    training_option = TrainingOption(training_option_val)
    base_model_name, training_config_file, multilot_config_file = _create_training_option_columns(training_option)

    config_col1, config_col2 = st.columns(2)
    training_config = _load_yaml_file(training_config_file, config_col1)
    multilot_config = _load_yaml_file(multilot_config_file, config_col2)
    site, tool, tech_layer, layer_group = _create_output_model_name_inputs()

    _start_training_job_button(
        training_option, base_model_name, site, tool, tech_layer, layer_group, training_config, multilot_config
    )

    st.divider()

    # _init_session_state_df()
    if "status_df_fin_current_page" not in st.session_state:
        st.session_state.status_df_fin_current_page = 1
    if "status_df_fin_page_size" not in st.session_state:
        st.session_state.status_df_fin_page_size = 10
    if "status_df_fin" not in st.session_state:
        st.session_state.status_df_fin = pd.DataFrame()
    if "detailed_df_fin" not in st.session_state:
        st.session_state.detailed_df_fin = pd.DataFrame()

    refresh_col, download_col, page_size_col, current_page_col = st.columns(4, vertical_alignment="bottom")

    with page_size_col:
        st.session_state.status_df_fin_page_size = st.number_input(
            "Page size", min_value=10, max_value=100, value=10, step=1
        )

    with current_page_col:
        st.session_state.status_df_fin_current_page = st.number_input("Page number", min_value=1, value=1, step=1)
        st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(
            page_size=st.session_state.status_df_fin_page_size, current_page=st.session_state.status_df_fin_current_page
        )

    with refresh_col:
        if st.button("Check all training jobs"):
            st.session_state.status_df_fin = api_helper.request_paginated_finetuning_status(
                page_size=st.session_state.status_df_fin_page_size,
                current_page=st.session_state.status_df_fin_current_page,
            )

    with download_col:
        st.download_button(
            label="Download training job statuses",
            data=st.session_state.status_df_fin.to_csv(index=False),
            file_name=f"training_status_{datetime.now().astimezone()}.csv",
            mime="text/csv",
        )

    progress_column = st.column_config.ProgressColumn(label="progress_bar", min_value=0, max_value=100)

    if st.session_state.status_df_fin.empty:
        st.write("")
    else:
        st.header("All training jobs")

    # Selection to find more detail
    if st.session_state.status_df_fin.empty:
        event_fin = None
        st.write("")
    else:
        event_fin = st.dataframe(
            st.session_state.status_df_fin,
            key="statuses_training",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={"progress": progress_column},
            height=ST_DATAFRAME_ROW_HEIGHT * (len(st.session_state.status_df_fin) + 1),
        )

    if event_fin and event_fin.selection:  # type: ignore[attr-defined] # "DataframeState" has no attribute "selection"
        if event_fin.selection["rows"]:  # type: ignore[attr-defined]
            # Get list of training_id for all selected fientuning jobs
            selected_finetuning_id = [
                st.session_state.status_df_fin.iloc[i]["training_id"]
                for i in event_fin.selection["rows"]  # type: ignore[attr-defined]
            ]

            # Get detailed statuses for each inference job and combine into one df
            raw_df_fin = api_helper.request_training_job_records(selected_finetuning_id)
            st.session_state.detailed_df_fin = raw_df_fin
            st.dataframe(st.session_state.detailed_df_fin, use_container_width=True)
            # Stop job button
            stop_job_button.gen(raw_df_fin)


def _create_training_option_columns(
    training_option: TrainingOption,
) -> tuple[str | None, UploadedFile | None, UploadedFile | None]:
    if training_option == TrainingOption.FINE_TUNE:
        base_model_name = st.selectbox(
            "Base model", options=api_helper.get_base_models(), index=0, format_func=helper.format_model_name
        )
    else:
        base_model_name = None

    yaml_help_text = """
    **Example of a valid .yaml config file:**\n
    data_paths:\n
    \\- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000\n
    \\- lot_id: N0_M0-0_20240101_000000\n
    &nbsp;&nbsp;lrf_path: /mnt/dbpc/xxx/N0_M0-0_20240101_000000_classified.lrf\n
    &nbsp;&nbsp;image_dir: /mnt/dbpc/xxx/N0_M0-0_20240101_000000/N0_M0-0_20240101_000000
    """
    r1_col1, r1_col2 = st.columns([1, 1])
    with r1_col1:
        training_config_file = st.file_uploader(f"Upload {training_option} Training Config (.yaml)", type=".yaml")
    with r1_col2:
        multilot_config_file = st.file_uploader("Upload Multi-lot Config (.yaml)", type=".yaml", help=yaml_help_text)

    return base_model_name, training_config_file, multilot_config_file


# TODO Make column type clearer
def _load_yaml_file(yaml_file: UploadedFile | None, col: Any) -> dict[str, Any]:
    loaded_yaml = {}
    if yaml_file is not None:
        loaded_yaml = yaml.load(yaml_file, Loader=yaml.Loader)
        # TODO: Validate yaml file format from backend and pass error message
        with col:
            st.json(loaded_yaml)

    return loaded_yaml


def _create_output_model_name_inputs() -> tuple[str, str, str, str]:
    r2_col1, r2_col2, r2_col3, r2_col4 = st.columns([1, 1, 2, 2])
    with r2_col1:
        site = st.text_input("Site", max_chars=20)
    with r2_col2:
        tool = st.text_input("Tool", value="x9u", max_chars=10)
    with r2_col3:
        tech_layer = st.text_input("Tech Layer", max_chars=50)
    with r2_col4:
        layer_group = st.text_input("Layer Group", max_chars=50)

    return site, tool, tech_layer, layer_group


def _missing_required_inputs(required_inputs: list[Any]) -> bool:
    for item in required_inputs:
        if not item:
            logger.error(
                "Missing user input detected. "
                "Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
            )
            st.error(
                "Missing user input detected. "
                "Please enter Site/Tool/Tech Layer/Layer Group, and upload a .yaml config file."
            )
            return True

    return False


def _start_training_job_button(
    training_option: TrainingOption,
    base_model_name: str | None,
    site: str,
    tool: str,
    tech_layer: str,
    layer_group: str,
    training_config: dict[str, Any],
    multilot_config: dict[str, Any],
) -> None:
    if st.button(f"Start {training_option} Job", type="primary"):
        # Validate user input first TODO: Should this be handled by backend?
        required_inputs = [site, tool, tech_layer, layer_group, multilot_config, training_config]
        if training_helper == TrainingOption.FINE_TUNE:
            required_inputs.append(base_model_name)

        if _missing_required_inputs(required_inputs):
            return

        output_model_name = training_helper.get_output_model_name(site, tool, tech_layer, layer_group)
        ff_core_training_request = FFCoreTrainingRequest(
            base_model_name=base_model_name,
            output_model_name=output_model_name,
            multilot_config=multilot_config,
            **training_config,
        )

        if training_option == TrainingOption.BASE_TRAIN:
            assert base_model_name is None
            response = api_helper.request_basetrain(ff_core_training_request)
        else:
            assert base_model_name is not None
            response = api_helper.request_finetune(ff_core_training_request)

        if response.json().get("status") == "error":
            code = response.json().get("code")
            message = response.json().get("message")
            st.text(f"Error code: {code}\nError message: {message}")
        else:
            training_id = response.json().get("training_id")
            st.text(f"Training Job ID: {training_id}")

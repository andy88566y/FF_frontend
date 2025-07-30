import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.subheader(body="Compile all true/false defects into a single data yaml file.")

    # data yaml file is required to get image_dir
    data_yaml_file = st.file_uploader("Upload data yaml (.yaml)", type=".yaml")
    if data_yaml_file is None:
        return

    data_yaml = yaml.load(data_yaml_file, Loader=yaml.Loader)
    st.caption(f"The uploaded data yaml contains {len(data_yaml['data_paths'])} lots.")

    st.divider()

    output_dir_col, output_lot_id_col = st.columns(2)
    with output_dir_col:
        output_dir = st.text_input(label="Output directory", placeholder="/mnt/xxx/")
    with output_lot_id_col:
        output_lot_id = st.text_input(label="Output Lot ID", placeholder="wXXX_true_defects")

    st.divider()

    # Options that apply for all modes
    mode_col, max_count_col = st.columns(2)
    with mode_col:
        compile_modes = api_helper.get_data_yaml_compile_modes().get("compile_modes", {})
        compile_mode = st.selectbox(label="Select compile mode", options=list(compile_modes.values()))
    with max_count_col:
        max_count = st.number_input(label="Maximum defect count", value=150)

    waive_pmode = st.toggle(label="Waive particle mode defects?", value=True)

    # Result directory is required to get defect probabilities
    result_dir = st.text_input(label="Result directory (Not required for All true / All false)")

    if st.button(label="Compile data yaml"):
        request = api_helper.compile_data_yaml(
            data_yaml_lots=data_yaml["data_paths"],
            output_dir=output_dir,
            output_lot_id=output_lot_id,
            compile_mode=compile_mode,
            max_count=max_count,
            waive_particle_mode=waive_pmode,
            result_dir=result_dir,
        )

        if request.get("status") == "error":
            message = request.get("message")
            st.error(f"{message}")
            return

        st.success(f"{request.get('message')}")

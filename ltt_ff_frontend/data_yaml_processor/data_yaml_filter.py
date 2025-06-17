from datetime import datetime

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, MaskType, ModelType, PixelSize, Site, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.subheader(body="Filter lots from a data yaml file.")

    data_yaml_file = st.file_uploader("Upload data yaml (.yaml)", type=".yaml")

    st.divider()

    st.subheader(body="[Optional] Upload a csv file containing Lot IDs to keep or remove from data yaml.")
    st.text("")
    filter_config_upload_col, keep_remove_col = st.columns(2)
    with filter_config_upload_col:
        lot_filter_config = st.file_uploader("Upload lot filter config (.csv)", type=".csv")
    with keep_remove_col:
        keep_filter_lots_selection = st.segmented_control(
            label="Keep or remove lots in lot filter config", options=["Keep lots", "Remove lots"]
        )
    with st.expander(label="Example .csv file"):
        example_df = pd.DataFrame(
            {"A": ["N0_M0-0_20240101_000000", "N0_M0-0_20240101_000001", "N0_M0-0_20240101_000002"]},
        )
        example_df.index = range(1, len(example_df) + 1)
        st.dataframe(data=example_df, hide_index=False, use_container_width=False)

    if lot_filter_config is not None:
        with st.expander(label="Filter Preview"):
            filter_df = pd.read_csv(lot_filter_config, header=None, names=["A"])
            filter_df.index = range(1, len(filter_df) + 1)
            st.dataframe(filter_df)

    st.divider()

    st.subheader(body="Select options to keep in the output data yaml.")
    st.text("If no options are selected for a certain category, all options will remain in the output data yaml.")
    st.text("E.g.: If no options are selected for Sites to keep, all sites will remain in the output data yaml.")
    wanted_sites = st.multiselect(label="Sites to keep in data yaml", options=[site.value for site in Site])
    wanted_tools = st.multiselect(label="Tools to keep in data yaml", options=[tool.value for tool in Tool])
    wanted_layer_groups = st.multiselect(
        label="Layer Groups to keep in data yaml", options=[lg.name for lg in LayerGroup]
    )
    wanted_pixel_size = st.multiselect(
        label="Pixel Sizes to keep in data yaml", options=[ps.value for ps in PixelSize if ps.value is not None]
    )
    wanted_mask_type = st.multiselect(label="Mask Types to keep in data yaml", options=[mt.value for mt in MaskType])

    if st.button(label="Prepare filtered data yaml", type="secondary"):
        if data_yaml_file is None:
            st.error("Please upload a data yaml file.")
            return

        filter_lots_list = filter_df.iloc[:, 0].tolist() if lot_filter_config is not None else []
        keep_lots = True if keep_filter_lots_selection == "Keep lots" else False

        data_yaml = yaml.load(data_yaml_file, Loader=yaml.Loader)

        request = api_helper.filter_data_yaml(
            original_data_yaml=data_yaml,
            filters={
                "site": wanted_sites,
                "tool": wanted_tools,
                "layer_group": wanted_layer_groups,
                "pixel_size": wanted_pixel_size,
                "mask_type": wanted_mask_type,
            },
            filter_lots_list=filter_lots_list,
            keep_lots=keep_lots,
        )

        if request.get("status") == "error":
            message = request.get("message")
            st.error(f"{message}")
            return

        st.success(f"{request.get('message')}")
        filtered_data_yaml = request["filtered_data_yaml"]

        st.download_button(
            label="Download data yaml file",
            data=yaml.dump(filtered_data_yaml),
            file_name=f"data_yaml_{datetime.now().astimezone()}.yaml",
            mime="text/yaml",
        )

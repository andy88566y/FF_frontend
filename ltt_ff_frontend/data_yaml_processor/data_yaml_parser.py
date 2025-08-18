from datetime import datetime

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import (
    ST_DATAFRAME_ROW_HEIGHT,
    LayerGroup,
    MaskType,
    ModelType,
    PixelSize,
    Site,
    TechLayer,
    Tool,
)
from ltt_ff_frontend.helpers import api_helper


def app() -> None:
    st.subheader(body="Check breakdown of lots in a data yaml.")

    data_yaml_file = st.file_uploader("Upload data yaml (.yaml)", type=".yaml")

    if data_yaml_file is None:
        return

    st.divider()

    data_yaml = yaml.load(data_yaml_file, Loader=yaml.Loader)

    request = api_helper.parse_data_yaml(data_yaml=data_yaml)

    if request.get("status") == "error":
        message = request.get("message")
        st.error(f"{message}")
        return

    st.success(f"{request.get('message')}")
    data_yaml_stats = request["data_yaml_stats"]

    formatted_stats = (
        pd.DataFrame.from_records(data_yaml_stats, columns=["LG", "SITE", "mask_type", "cnt"])
        .pivot(index=["LG", "SITE"], columns="mask_type", values="cnt")
        .fillna(0)
        .astype("int32")
    )

    st.dataframe(
        data=formatted_stats,
        height=ST_DATAFRAME_ROW_HEIGHT * (len(formatted_stats) + 2),
    )

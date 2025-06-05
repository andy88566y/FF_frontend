from datetime import date

import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, ModelType, PixelSize, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper


def app() -> None:
    model_week_col, model_list_col = st.columns([3, 7])
    with model_week_col:
        ds_model_dirs = api_helper.get_ds_model_dirs()
        model_week = st.selectbox(label="Filter by week", options=ds_model_dirs, key="model_week")
    with model_list_col:
        ds_models = api_helper.get_ds_models(model_week)
        model_to_convert = st.selectbox(
            label="Model to convert", options=ds_models, format_func=helper.format_model_name, key="model_to_convert"
        )

    (
        channel_size_title_col,
        channel_size_col_1,
        channel_size_col_2,
        channel_size_col_3,
        _,
        kernel_size_title_col,
        kernel_size_col_1,
        kernel_size_col_2,
        kernel_size_col_3,
    ) = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])

    with channel_size_title_col:
        st.text("Channel size:")
    with channel_size_col_1:
        channel_size_1 = st.text_input(
            label="channel_size_1", value="", label_visibility="collapsed", key="channel_size_1"
        )

    with channel_size_col_2:
        channel_size_2 = st.text_input(
            label="channel_size_2", value="", label_visibility="collapsed", key="channel_size_2"
        )

    with channel_size_col_3:
        channel_size_3 = st.text_input(
            label="channel_size_3", value="", label_visibility="collapsed", key="channel_size_3"
        )

    with kernel_size_title_col:
        st.text("Kernel size:")
    with kernel_size_col_1:
        kernel_size_1 = st.text_input(
            label="kernel_size_1", value="", label_visibility="collapsed", key="kernel_size_1"
        )

    with kernel_size_col_2:
        kernel_size_2 = st.text_input(
            label="kernel_size_2", value="", label_visibility="collapsed", key="kernel_size_2"
        )

    with kernel_size_col_3:
        kernel_size_3 = st.text_input(
            label="kernel_size_3", value="", label_visibility="collapsed", key="kernel_size_3"
        )

    th_col, cr_col = st.columns(2)

    with th_col:
        model_threshold = st.number_input(
            label="Model threshold",
            value=0.5,
            step=1e-5,
            format="%.5f",
            help="Probabilities below threshold will be considered as non-defects.",
        )

    with cr_col:
        cr_1_specificity = st.number_input(
            label="CR1/specificity",
            value=0.5,
            step=1e-4,
            format="%.4f",
            help="False Filter Rate when Capture Rate is 100%.",
        )

    assign_threshold_c = st.toggle(label="Assign Model Threshold_c?")
    if assign_threshold_c:
        with st.expander("Threshold_c"):
            thc_col, thc_cr_col, thc_ffr_col = st.columns(3)
        with thc_col:
            model_threshold_c = st.number_input(
                label="Model threshold_c",
                value=0.0,
                step=1e-5,
                format="%.5f",
            )
        with thc_cr_col:
            thc_cr = st.number_input(
                label="THC/CR",
                value=0.0,
                step=1e-4,
                format="%.4f",
            )
        with thc_ffr_col:
            thc_ffr = st.number_input(
                label="THC/FFR",
                value=0.0,
                step=1e-4,
                format="%.4f",
            )
    else:
        model_threshold_c = None
        thc_cr = None
        thc_ffr = None

    tool_col, tl_col, lg_col, ps_col, mt_col = st.columns(5)

    with tool_col:
        tool = st.selectbox(label="Tool", options=[tool.value for tool in Tool], key="tool_selectbox")

    with tl_col:
        tech_layer = st.selectbox(label="Tech Layer", options=[tl.value for tl in TechLayer], key="tl_selectbox")

    with lg_col:
        layer_group = st.selectbox(label="Layer Group", options=[lg.name for lg in LayerGroup], key="lg_selectbox")

    with ps_col:
        pixel_size = st.selectbox(label="Pixel Size", options=[ps.value for ps in PixelSize], key="ps_selectbox")

    with mt_col:
        model_type = st.selectbox(label="Model Type", options=[mt.value for mt in ModelType], key="mt_selectbox")

    conversion_date = st.date_input("Date of conversion", value=date.today())

    if st.button(label="Convert model", type="primary"):
        # Quick check for any missing user input
        for param_name, param_value in st.session_state.items():
            if not param_value:
                logger.error(f"Missing user input for {param_name}! Model conversion failed to start.")
                st.error(f"Missing user input for {param_name}! Model conversion failed to start.")
                return

        # Check invalid model threshold or cr_1_specificity
        if not 0.0 <= model_threshold <= 1.0 or not 0.0 <= cr_1_specificity <= 1.0:
            logger.error(
                f"Model threshold [{model_threshold}] and CR1/Specificity [{cr_1_specificity}] must be between 0 and 1!"
                "\nModel conversion failed to start."
            )
            st.error(
                f"Model threshold [{model_threshold}] and CR1/Specificity [{cr_1_specificity}] must be between 0 and 1!"
                "\n\nModel conversion failed to start."
            )
            return

        # Check invalid threshold_c, THC/CR, and THC/FFR
        if assign_threshold_c:
            if not 0.0 <= model_threshold_c <= 1.0 or not 0.0 <= thc_cr <= 1.0 or not 0.0 <= thc_ffr <= 1.0:
                logger.error(
                    f"Model threshold_c [{model_threshold_c}], THC/CR [{thc_cr}] and THC/FFR [{thc_ffr}] must be between 0 and 1!"
                    "\nModel conversion failed to start."
                )
                st.error(
                    f"Model threshold_c [{model_threshold_c}], THC/CR [{thc_cr}] and THC/FFR [{thc_ffr}] must be between 0 and 1!"
                    "\nModel conversion failed to start."
                )
                return

        model_details = {
            "model_architecture": "DUALSTREAMCNN",
            "model_to_convert": model_to_convert,
            "channel_size": [int(channel_size_1), int(channel_size_2), int(channel_size_3)],
            "kernel_size": [int(kernel_size_1), int(kernel_size_2), int(kernel_size_3)],
            "model_threshold": model_threshold,
            "model_threshold_c": model_threshold_c,
            "cr_1_specificity": cr_1_specificity,
            "thc_cr": thc_cr,
            "thc_ffr": thc_ffr,
            "tool": tool,
            "tech_layer": tech_layer,
            "layer_group": layer_group,
            "pixel_size": pixel_size,
            "model_type": model_type,
            "date": f"{conversion_date.year}{conversion_date.month:02d}{conversion_date.day:02d}",
        }

        r = api_helper.convert_model(model_details)

        if r["status"] == "completed":
            st.success(r["message"])
        else:
            st.error(r["message"])

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

    ###################################################################################################################
    # Feat channel size, feat kernel size                                                                             #
    ###################################################################################################################
    (
        feat_channel_size_title_col,
        feat_channel_size_col_1,
        feat_channel_size_col_2,
        feat_channel_size_col_3,
        _,
        feat_kernel_size_title_col,
        feat_kernel_size_col_1,
        feat_kernel_size_col_2,
        feat_kernel_size_col_3,
    ) = st.columns([2, 2, 2, 2, 1, 2, 2, 2, 2])

    with feat_channel_size_title_col:
        st.text("Feat Channel size:")
    with feat_channel_size_col_1:
        feat_channel_size_1 = st.text_input(
            label="feat_channel_size_1", value="", label_visibility="collapsed", key="feat_channel_size_1"
        )

    with feat_channel_size_col_2:
        feat_channel_size_2 = st.text_input(
            label="feat_channel_size_2", value="", label_visibility="collapsed", key="feat_channel_size_2"
        )

    with feat_channel_size_col_3:
        feat_channel_size_3 = st.text_input(
            label="feat_channel_size_3", value="", label_visibility="collapsed", key="feat_channel_size_3"
        )

    with feat_kernel_size_title_col:
        st.text("Feat Kernel size:")
    with feat_kernel_size_col_1:
        feat_kernel_size_1 = st.text_input(
            label="feat_kernel_size_1", value="", label_visibility="collapsed", key="feat_kernel_size_1"
        )

    with feat_kernel_size_col_2:
        feat_kernel_size_2 = st.text_input(
            label="feat_kernel_size_2", value="", label_visibility="collapsed", key="feat_kernel_size_2"
        )

    with feat_kernel_size_col_3:
        feat_kernel_size_3 = st.text_input(
            label="feat_kernel_size_3", value="", label_visibility="collapsed", key="feat_kernel_size_3"
        )

    overwrite_default_params = st.toggle(
        label="Overwrite defaults for additional params", value=False, key="use_default_params"
    )

    if overwrite_default_params:
        with st.expander(label="Additional params"):
            ############################################################################################################
            # Feat stride, feat padding                                                                                #
            ############################################################################################################
            (
                feat_stride_title_col,
                feat_stride_col_1,
                feat_stride_col_2,
                feat_stride_col_3,
                _,
                feat_padding_title_col,
                feat_padding_col_1,
                feat_padding_col_2,
                feat_padding_col_3,
            ) = st.columns([2, 2, 2, 2, 1, 2, 2, 2, 2])

            with feat_stride_title_col:
                st.text("Feat Stride:")
            with feat_stride_col_1:
                feat_stride_1 = st.text_input(
                    label="feat_stride_1", value="", label_visibility="collapsed", key="feat_stride_1"
                )

            with feat_stride_col_2:
                feat_stride_2 = st.text_input(
                    label="feat_stride_2", value="", label_visibility="collapsed", key="feat_stride_2"
                )

            with feat_stride_col_3:
                feat_stride_3 = st.text_input(
                    label="feat_stride_3", value="", label_visibility="collapsed", key="feat_stride_3"
                )

            with feat_padding_title_col:
                st.text("Feat Padding:")
            with feat_padding_col_1:
                feat_padding_1 = st.text_input(
                    label="feat_padding_1", value="", label_visibility="collapsed", key="feat_padding_1"
                )

            with feat_padding_col_2:
                feat_padding_2 = st.text_input(
                    label="feat_padding_2", value="", label_visibility="collapsed", key="feat_padding_2"
                )

            with feat_padding_col_3:
                feat_padding_3 = st.text_input(
                    label="feat_padding_3", value="", label_visibility="collapsed", key="feat_padding_3"
                )

            ############################################################################################################
            # Feat use pooling, feat pool kernel, feat pool stride                                                     #
            ############################################################################################################
            (
                feat_use_pooling_title_col,
                feat_use_pooling_col_1,
                feat_use_pooling_col_2,
                feat_use_pooling_col_3,
                _,
                feat_pool_kernel_title_col,
                feat_pool_kernel_col,
                feat_pool_stride_title_col,
                feat_pool_stride_col,
            ) = st.columns([2, 2, 2, 2, 1, 2, 2, 2, 2])

            with feat_use_pooling_title_col:
                st.text("Feat Use Pooling:")
            with feat_use_pooling_col_1:
                feat_use_pooling_1 = st.selectbox(
                    label="feat_use_pooling_1",
                    options=[True, False],
                    label_visibility="collapsed",
                    key="feat_use_pooling_1",
                )

            with feat_use_pooling_col_2:
                feat_use_pooling_2 = st.selectbox(
                    label="feat_use_pooling_2",
                    options=[True, False],
                    label_visibility="collapsed",
                    key="feat_use_pooling_2",
                )

            with feat_use_pooling_col_3:
                feat_use_pooling_3 = st.selectbox(
                    label="feat_use_pooling_3",
                    options=[True, False],
                    label_visibility="collapsed",
                    key="feat_use_pooling_3",
                )

            with feat_pool_kernel_title_col:
                st.text("Feat Pool Kernel:")
            with feat_pool_kernel_col:
                feat_pool_kernel = st.text_input(
                    label="feat_pool_kernel", value="", label_visibility="collapsed", key="feat_pool_kernel"
                )

            with feat_pool_stride_title_col:
                st.text("Feat Pool Stride:")
            with feat_pool_stride_col:
                feat_pool_stride = st.text_input(
                    label="feat_pool_stride", value="", label_visibility="collapsed", key="feat_pool_stride"
                )

            ############################################################################################################
            # Diff channels, diff kernels                                                                              #
            ############################################################################################################
            (
                diff_channels_title_col,
                diff_channels_col_1,
                diff_channels_col_2,
                _,
                diff_kernels_title_col,
                diff_kernels_col_1,
                diff_kernels_col_2,
            ) = st.columns([2, 2, 2, 1, 2, 2, 2])

            with diff_channels_title_col:
                st.text("Diff Channels:")
            with diff_channels_col_1:
                diff_channels_1 = st.text_input(
                    label="diff_channels_1", value="", label_visibility="collapsed", key="diff_channels_1"
                )

            with diff_channels_col_2:
                diff_channels_2 = st.text_input(
                    label="diff_channels_2", value="", label_visibility="collapsed", key="diff_channels_2"
                )

            with diff_kernels_title_col:
                st.text("Diff Kernels:")
            with diff_kernels_col_1:
                diff_kernels_1 = st.text_input(
                    label="diff_kernels_1", value="", label_visibility="collapsed", key="diff_kernels_1"
                )

            with diff_kernels_col_2:
                diff_kernels_2 = st.text_input(
                    label="diff_kernels_2", value="", label_visibility="collapsed", key="diff_kernels_2"
                )

            ############################################################################################################
            # Diff stride, diff padding                                                                                #
            ############################################################################################################
            (
                diff_stride_title_col,
                diff_stride_col_1,
                diff_stride_col_2,
                _,
                diff_padding_title_col,
                diff_padding_col_1,
                diff_padding_col_2,
            ) = st.columns([2, 2, 2, 1, 2, 2, 2])

            with diff_stride_title_col:
                st.text("Diff Stride:")
            with diff_stride_col_1:
                diff_stride_1 = st.text_input(
                    label="diff_stride_1", value="", label_visibility="collapsed", key="diff_stride_1"
                )

            with diff_stride_col_2:
                diff_stride_2 = st.text_input(
                    label="diff_stride_2", value="", label_visibility="collapsed", key="diff_stride_2"
                )

            with diff_padding_title_col:
                st.text("Diff Padding:")
            with diff_padding_col_1:
                diff_padding_1 = st.text_input(
                    label="diff_padding_1", value="", label_visibility="collapsed", key="diff_padding_1"
                )

            with diff_padding_col_2:
                diff_padding_2 = st.text_input(
                    label="diff_padding_2", value="", label_visibility="collapsed", key="diff_padding_2"
                )

            ############################################################################################################
            # Diff use pooling                                                                                         #
            ############################################################################################################
            (
                diff_use_pooling_title_col,
                diff_use_pooling_col_1,
                diff_use_pooling_col_2,
                _,
            ) = st.columns([2, 2, 2, 7])

            with diff_use_pooling_title_col:
                st.text("Diff Use Pooling:")
            with diff_use_pooling_col_1:
                diff_use_pooling_1 = st.selectbox(
                    label="diff_use_pooling_1",
                    options=[True, False],
                    label_visibility="collapsed",
                    key="diff_use_pooling_1",
                )

            with diff_use_pooling_col_2:
                diff_use_pooling_2 = st.selectbox(
                    label="diff_use_pooling_2",
                    options=[True, False],
                    label_visibility="collapsed",
                    key="diff_use_pooling_2",
                )

            ############################################################################################################
            # Classifier hidden, classifier dropout                                                                    #
            ############################################################################################################
            (
                classifier_hidden_title_col,
                classifier_hidden_col_1,
                classifier_hidden_col_2,
                _,
                classifier_dropout_title_col,
                classifier_dropout_col,
                _,
            ) = st.columns([2, 2, 2, 1, 2, 2, 2])

            with classifier_hidden_title_col:
                st.text("Classifier Hidden:")
            with classifier_hidden_col_1:
                classifier_hidden_1 = st.text_input(
                    label="classifier_hidden_1", value="", label_visibility="collapsed", key="classifier_hidden_1"
                )

            with classifier_hidden_col_2:
                classifier_hidden_2 = st.text_input(
                    label="classifier_hidden_2", value="", label_visibility="collapsed", key="classifier_hidden_2"
                )

            with classifier_dropout_title_col:
                st.text("Classifier Dropout:")
            with classifier_dropout_col:
                classifier_dropout = st.text_input(
                    label="classifier_dropout", value="", label_visibility="collapsed", key="classifier_dropout"
                )
    else:
        feat_stride_1, feat_stride_2, feat_stride_3 = "2", "2", "1"
        feat_padding_1, feat_padding_2, feat_padding_3 = "3", "2", "1"
        feat_use_pooling_1, feat_use_pooling_2, feat_use_pooling_3 = True, False, False
        feat_pool_kernel = "2"
        feat_pool_stride = "2"
        diff_channels_1, diff_channels_2 = "256", "128"
        diff_kernels_1, diff_kernels_2 = "3", "3"
        diff_stride_1, diff_stride_2 = "1", "1"
        diff_padding_1, diff_padding_2 = "1", "1"
        diff_use_pooling_1, diff_use_pooling_2 = True, True
        classifier_hidden_1, classifier_hidden_2 = "512", "256"
        classifier_dropout = "0.5"

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
            if (isinstance(param_value, str) and len(param_value) < 1) or (
                isinstance(param_value, bool) and param_value is None
            ):
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
            "model_architecture": "DUALSTREAMCNN_V2",
            "model_to_convert": model_to_convert,
            "feat_channels": (int(feat_channel_size_1), int(feat_channel_size_2), int(feat_channel_size_3)),
            "feat_kernels": (int(feat_kernel_size_1), int(feat_kernel_size_2), int(feat_kernel_size_3)),
            "feat_stride": (int(feat_stride_1), int(feat_stride_2), int(feat_stride_3)),
            "feat_padding": (int(feat_padding_1), int(feat_padding_2), int(feat_padding_3)),
            "feat_use_pooling": (feat_use_pooling_1, feat_use_pooling_2, feat_use_pooling_3),
            "feat_pool_kernel": int(feat_pool_kernel),
            "feat_pool_stride": int(feat_pool_stride),
            "diff_channels": (int(diff_channels_1), int(diff_channels_2)),
            "diff_kernels": (int(diff_kernels_1), int(diff_kernels_2)),
            "diff_stride": (int(diff_stride_1), int(diff_stride_2)),
            "diff_padding": (int(diff_padding_1), int(diff_padding_2)),
            "diff_use_pooling": (diff_use_pooling_1, diff_use_pooling_2),
            "classifier_hidden": (int(classifier_hidden_1), int(classifier_hidden_2)),
            "classifier_dropout": float(classifier_dropout),
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

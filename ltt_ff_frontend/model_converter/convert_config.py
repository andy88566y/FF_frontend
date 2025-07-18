from datetime import date
from typing import Any

import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.constant import LayerGroup, ModelType, PixelSize, TechLayer, Tool
from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.shared_components import helper


class FlowList(list):
    """
    A subclass of list to signal flow-style YAML representation.
    """


def flow_list_representer(dumper: yaml.representer.Representer, data: FlowList) -> yaml.nodes.SequenceNode:
    """
    Custom representer to serialize FlowList in flow style (i.e. show lists in [x, x, x] style)

    tag:yaml.org,2002 is the namespace for YAML's core schema
    seq stands for sequence, which is an ordered collection of items, like Python's list
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


config_previews = {
    "DualStreamCNN": {
        "models": [
            {
                "input_model_path": "20250424/model_N2_CUT_45_20250420T161109Z_02b46747.ckpt",
                "model_params": {
                    "model_type": "DualStreamCNN",
                    "model_init_params": {
                        "input_size": 256,
                        "max_refs": 4,
                        "channel_size": FlowList([32, 64, 128]),
                        "kernel_size": FlowList([3, 3, 5]),
                    },
                    "model_threshold": 0.0,
                    "model_threshold_c": None,
                },
                "model_metadata": {"metrics": {"CR1/specificity": 0.0}},
                "techlayer": "N2",
                "layergroup": "CUT",
                "pixelsize": 45,
                "modeltype": "base",
                "datetime_iso": "20250424T000000Z",
                "tool": "x9u",
            }
        ]
    },
    "dual_stream_cnn_v2": {
        "models": [
            {
                "input_model_path": "20250710/model_N2_PO_45_20250708t005926z_dualp.ckpt",
                "model_params": {
                    "model_type": "dual_stream_cnn_v2",
                    "model_init_params": {
                        "input_size": 256,
                        "max_refs": 4,
                        "feat_channels": FlowList([128, 256, 512]),
                        "feat_kernels": FlowList([5, 3, 1]),
                        "feat_stride": FlowList([2, 2, 1]),
                        "feat_padding": FlowList([3, 2, 1]),
                        "feat_use_pooling": FlowList([True, False, False]),
                        "feat_pool_kernel": 2,
                        "feat_pool_stride": 2,
                        "diff_channels": FlowList([256, 128]),
                        "diff_kernels": FlowList([3, 3]),
                        "diff_stride": FlowList([1, 1]),
                        "diff_padding": FlowList([1, 1]),
                        "diff_use_pooling": FlowList([True, True]),
                        "classifier_hidden": FlowList([512, 256]),
                        "classifier_dropout": 0.5,
                    },
                    "model_threshold": 0.0,
                    "model_threshold_c": None,
                },
                "model_metadata": {"metrics": {"CR1/specificity": 0.0, "THC/CR": 1.0, "THC/specificity": 0.0}},
                "techlayer": "N2",
                "layergroup": "CUT",
                "pixelsize": 45,
                "modeltype": "base",
                "datetime_iso": "20250424T000000Z",
                "tool": "x9u",
            }
        ]
    },
    "timm_classification": {
        "models": [
            {
                "input_model_path": "20250710/model_N5_M0M2_45_20250710T000000Z_23039905.ckpt",
                "model_params": {
                    "model_type": "timm_classification",
                    "model_init_params": {
                        "input_size": FlowList([2, 256, 256]),
                        "alignment_mode": "UP",
                        "normalization_mode": "NOP",
                        "diff_mode": "WD_1",
                        "backbone": "tf_efficientnetv2_s",
                        "fc_hidden_dims": FlowList([256, 32]),
                        "fc_dropout": 0.5,
                        "use_conv3d": True,
                        "conv3d_out_channels": 32,
                        "conv3d_kernel_size": FlowList([2, 3, 3]),
                    },
                    "model_threshold": 0.0,
                    "model_threshold_c": 0.5,
                },
                "model_metadata": {"metrics": {"CR1/specificity": 0.0, "THC/CR": 0.9925, "THC/specificity": 0.9978}},
                "techlayer": "N2",
                "layergroup": "CUT",
                "pixelsize": 45,
                "modeltype": "BASE",
                "datetime_iso": "20250424T000000Z",
                "tool": "x9u",
            },
            {
                "input_model_path": "20250710/model_N5_M0M2_45_20250710T000000Z_23039905.ckpt",
                "model_params": {
                    "model_type": "timm_classification",
                    "model_init_params": {
                        "input_size": FlowList([2, 256, 256]),
                        "alignment_mode": "UP",
                        "normalization_mode": "NOP",
                        "diff_mode": "WD_1",
                        "backbone": "tf_efficientnetv2_s",
                        "fc_hidden_dims": FlowList([256, 32]),
                        "fc_dropout": 0.5,
                    },
                    "model_threshold": 0.0,
                    "model_threshold_c": 0.5,
                },
                "model_metadata": {"metrics": {"CR1/specificity": 0.0, "THC/CR": 0.9925, "THC/specificity": 0.9978}},
                "techlayer": "N2",
                "layergroup": "CUT",
                "pixelsize": 45,
                "modeltype": "S1",
                "datetime_iso": "20250424T000000Z",
                "tool": "x9u",
            },
        ]
    },
}


def app() -> None:
    # Upload config file
    conversion_config_file = st.file_uploader("Upload Model Conversion Config File (.yaml)", type=".yaml")

    yaml.add_representer(FlowList, flow_list_representer)

    if conversion_config_file is not None:
        model_conversion_config = yaml.load(conversion_config_file, Loader=yaml.Loader)

        with st.expander(label="Model Conversion Config preview"):
            # Show lists with inline brackets instead by wrapping lists in a FlowList
            for model_config in model_conversion_config["models"]:
                for k, v in model_config["model_params"]["model_init_params"].items():
                    if isinstance(v, list):
                        model_config["model_params"]["model_init_params"][k] = FlowList(v)
            st.code(yaml.dump(data=model_conversion_config, sort_keys=False), language="yaml")

    with st.expander(label="Model Conversion Config File examples"):
        st.caption("Convert multiple models using one config file (see timm_classification example).")
        st.caption("A config file can have various model architectures (e.g. DualStream + timm_classification).")
        preview_selection = st.segmented_control(
            label="Preview Selection",
            options=["DualStreamCNN", "dual_stream_cnn_v2", "timm_classification"],
            selection_mode="single",
            default="DualStreamCNN",
            label_visibility="collapsed",
        )
        if preview_selection is not None:
            st.code(yaml.dump(data=config_previews[preview_selection], sort_keys=False), language="yaml")

    if st.button(label="Convert model", type="primary"):
        st.caption("Model conversion in progress. When completed, a success message will be shown below.")
        r = api_helper.convert_model(model_conversion_config=model_conversion_config)

        if r["status"] == "completed":
            st.success(r["message"])
            logger.success(r["message"])
        else:
            st.error(r["message"])
            logger.error(r["message"])

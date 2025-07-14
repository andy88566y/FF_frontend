import itertools
import os
from collections import defaultdict

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper
from ltt_ff_frontend.regression_test_ui.regression_result_utils import (
    RULE_TABLE_COLUMNS,
    TABLE_COLUMNS,
    check_new_dir,
    get_holdout_layer_filter,
    get_summary,
    get_summary_per_site,
)


RUN_MODES = ["Normal", "with Rule", "Rule only", "Holdout"]
CONFIG_NAME = "regression_config.yaml"


def app() -> None:
    logger.debug("Loading Regression Result Viewer...")
    st.title("Regression Test Result Viewer")
    st.caption("Visualize Regression Test Result")

    r1_col1, r1_col2 = st.columns([1, 1])
    output_dir_default = "/mnt/fs0/MLE/ff_docker_output/mle_regression_test"
    with r1_col1:
        regression_result_parent_dir = st.text_input("Regression Result Directory", value=output_dir_default)
    with r1_col2:
        specific_dir = st.selectbox("Choose Date", sorted(os.listdir(regression_result_parent_dir), reverse=True))
        if regression_result_parent_dir and specific_dir:
            regression_result_dir = os.path.join(regression_result_parent_dir, specific_dir)
        check_new_dir(regression_result_dir)
        yaml_path = os.path.join(regression_result_parent_dir, CONFIG_NAME)
        data = None
        if os.path.isfile(yaml_path):
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        layers = data["layers"] if data else ["OD", "PO", "CUT", "M0M2", "M1", "VIA"]
        sites = data["sites"] if data else ["F20", "F12", "F18A", "F18B", "F18EBO", "F15EBO"]
        p_week, c_week = data["weeks"] if data else (None, None)

    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        previous_week = st.text_input("Previous Week", value=p_week)
    with r2_col2:
        current_week = st.text_input("Current Week", value=c_week)

    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        layer_filter = st.multiselect("View Layers", layers, default=layers)

    with r3_col2:
        site_filter = st.multiselect("View Sites", sites, default=sites)

    layer_site_filter = {f"{layer}_{site}" for layer in layer_filter for site in site_filter}
    mode = st.segmented_control("Run Mode", RUN_MODES, default="Normal")

    if mode is None:
        st.error("Run Mode Option can not be None!")
        return

    if previous_week and current_week and regression_result_dir:
        weeks = [f"W{previous_week}", f"W{current_week}"]
        rule_weeks = [f"W{current_week}"]
    else:
        return

    regression_testcases = defaultdict(list)
    for _dir in os.listdir(regression_result_dir):
        key = "_".join(_dir.split("_")[:3])
        regression_testcases[key].append(_dir)

    if mode == "Normal":
        if "normal" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(layer_site_filter, weeks)]
            st.session_state.normal = api_helper.fetch_regression_result(
                regression_testcases, prefixes, regression_result_dir, mode
            )["regression_result"]
        headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in weeks]
        summary = get_summary(layer_filter, st.session_state.normal, weeks)
        summary_p_site, layer_p_site = get_summary_per_site(layer_filter, st.session_state.normal, TABLE_COLUMNS, weeks)
        st.subheader("Regression Results Summary")
        st.dataframe(pd.DataFrame(summary, columns=headers, index=layer_filter))
        st.subheader("Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_p_site, columns=headers, index=layer_p_site))

    elif mode == "with Rule":
        if "rule" not in st.session_state:
            st.session_state.rule = {"regression_result": defaultdict(list), "ffr_check_memo": defaultdict()}
            for m in ["Normal", "with Rule"]:
                prefixes = [f"{w}_{lf}" for lf, w in itertools.product(layer_site_filter, rule_weeks)]
                result = api_helper.fetch_regression_result(regression_testcases, prefixes, regression_result_dir, m)
                st.session_state.rule["regression_result"].update(result["regression_result"])
                st.session_state.rule["ffr_check_memo"].update(result["ffr_check_memo"])
        regression_result = st.session_state.rule["regression_result"]
        ffr_check_memo = st.session_state.rule["ffr_check_memo"]
        headers_weeks = [rule_weeks[0], f"{rule_weeks[0]}with Rule"]
        headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in headers_weeks]
        summary_p_site, layer_p_site = get_summary_per_site(
            layer_filter, regression_result, TABLE_COLUMNS, headers_weeks
        )
        st.subheader("RULE Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_p_site, columns=headers, index=layer_p_site))
        # list out ffr drop > 5% lots
        warnings = []

        for layer_site in layer_site_filter:
            original_key = f"{headers_weeks[0]}_{layer_site}"
            with_rule_key = f"{headers_weeks[1]}_{layer_site}"
            for mask_type in ["flush", "production"]:
                memo = ffr_check_memo.get(f"{with_rule_key}_{mask_type}", {})
                for lot_id, rule_ffr in memo.items():
                    no_rule_ffr = ffr_check_memo[f"{original_key}_{mask_type}"][lot_id]
                    if rule_ffr > no_rule_ffr:
                        warnings.append(
                            {
                                "LayerFab": layer_site,
                                "MaskType": mask_type,
                                "LotID": lot_id,
                                "NoRuleFFR": no_rule_ffr,
                                "RuleFFR": rule_ffr,
                                "Issue": "FFR increased after applying RULE",
                            }
                        )
                    elif no_rule_ffr - rule_ffr >= 0.05:
                        warnings.append(
                            {
                                "LayerFab": layer_site,
                                "MaskType": mask_type,
                                "LotID": lot_id,
                                "NoRuleFFR": no_rule_ffr,
                                "RuleFFR": rule_ffr,
                                "Issue": "FFR decreased > 5% after applying RULE",
                            }
                        )
        if len(warnings) != 0:
            st.warning("⚠️ Some lots show unexpected FFR changes after applying RULE Model.")
            st.dataframe(pd.DataFrame(warnings))

    elif mode == "Rule only":
        if "r_only" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(layer_site_filter, rule_weeks)]
            st.session_state.r_only = api_helper.fetch_regression_result(
                regression_testcases, prefixes, regression_result_dir, mode
            )["regression_result"]
        headers_weeks = [f"{week}{mode}" for week in rule_weeks]
        summary_p_site, layer_p_site = get_summary_per_site(
            layer_filter, st.session_state.r_only, RULE_TABLE_COLUMNS, headers_weeks
        )
        headers = [f"{week} {col}" for col in RULE_TABLE_COLUMNS for week in headers_weeks]
        st.subheader("RULE ONLY Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_p_site, columns=headers, index=layer_p_site))

    elif mode == "Holdout":
        holdout_test_cases = defaultdict(list)
        for _dir in os.listdir(regression_result_dir + "_holdout"):
            key = "_".join(_dir.split("_")[:3])
            holdout_test_cases[key].append(_dir)
        if "holdout" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(layer_site_filter, weeks)]
            st.session_state.holdout = api_helper.fetch_regression_result(
                holdout_test_cases, prefixes, regression_result_dir + "_holdout", "Normal"
            )["regression_result"]
        holdout_layer_filter = get_holdout_layer_filter(
            layer_filter, st.session_state.holdout, weeks[0], TABLE_COLUMNS[0]
        )
        headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in weeks]
        summary = get_summary(holdout_layer_filter, st.session_state.holdout, weeks)
        st.subheader("Holdout Regression Results Summary")
        st.dataframe(pd.DataFrame(summary, columns=headers, index=holdout_layer_filter))
        site_headers = [f"{week} {col}" for col in RULE_TABLE_COLUMNS for week in weeks]
        summary_per_site, layer_per_site = get_summary_per_site(
            holdout_layer_filter, st.session_state.holdout, RULE_TABLE_COLUMNS, weeks
        )
        st.subheader("Holdout Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_per_site, columns=site_headers, index=layer_per_site))

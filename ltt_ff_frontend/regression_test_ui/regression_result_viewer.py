import itertools
import os
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from ltt_ff_frontend.helpers import api_helper


CR_COL = "(CR OOS lots)"
FFR_COL = "(FFR OOS lots) Exclude flush"
AVG_FFR_COL = "avg ffr"
TABLE_COLUMNS = [CR_COL, FFR_COL]
RULE_TABLE_COLUMNS = [CR_COL, FFR_COL, AVG_FFR_COL]
RUN_MODES = ["Normal", "with Rule", "Rule only", "Holdout"]
CONFIG_NAME = "regression_config.yaml"


def check_new_dir(result_dir: str) -> None:
    if "regression_result_dir" not in st.session_state or result_dir != st.session_state.regression_result_dir:
        logger.info("new result dir detected, clear old session_state value")
        for key in [
            "normal",
            "rule",
            "r_only",
            "holdout",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.regression_result_dir = result_dir


def parse_cr_value(n1: int, prev_n1: int, d: int) -> tuple[str, int]:
    mark = "🔺" if n1 > prev_n1 else ""
    return f"{mark}{n1}/{d}", n1


def parse_ffr_value(n1: int, n2: int, prev_n1: int, prev_n2: int, d: int) -> tuple[str, int, int]:
    mark1 = "🔺" if n1 > prev_n1 else ""
    mark2 = "🔸" if n2 > prev_n2 else ""
    return f"{mark1}{n1}({mark2}{n2})/{d}", n1, n2


def get_summary(layer_filter: list[str], results: dict[str, list], weeks: list) -> list:
    summary = []
    for layer in layer_filter:
        row = []
        for col in TABLE_COLUMNS:
            prev_n1 = prev_n2 = 9999
            for week in weeks:
                total_n1, total_n2, denominator = 0, 0, 0
                for info in results[f"{week}#{layer}#{col}"]:
                    n, d = info.split()[-1].split("/")
                    denominator += int(d)
                    if col == CR_COL:
                        total_n1 += int(n)
                    else:
                        assert col == FFR_COL
                        n1, n2 = map(int, n[:-1].split("("))
                        total_n1 += n1
                        total_n2 += n2
                if col == CR_COL:
                    formatted, prev_n1 = parse_cr_value(total_n1, prev_n1, denominator)
                else:
                    assert col == FFR_COL
                    formatted, prev_n1, prev_n2 = parse_ffr_value(total_n1, total_n2, prev_n1, prev_n2, denominator)
                row.append(formatted)

        summary.append(row)
    return summary


def get_summary_p_site(layer_filter: list[str], results: dict[str, list], columns: list, weeks: list) -> tuple:
    summary_per_site = []
    layer_per_site = []

    for layer in layer_filter:
        site_len = len(results[f"{weeks[0]}#{layer}#{columns[0]}"])
        for i in range(site_len):
            row = []
            for col in columns:
                prev_n1 = prev_n2 = 9999
                for week in weeks:
                    site, info = results[f"{week}#{layer}#{col}"][i].split()
                    if "/" not in info:
                        row.append(results[f"{week}#{layer}#{col}"][i])
                        continue
                    n, d = info.split("/")
                    if col == CR_COL:
                        formatted, prev_n1 = parse_cr_value(int(n), prev_n1, d)
                    elif col == FFR_COL:
                        n1, n2 = map(int, n[:-1].split("("))
                        formatted, prev_n1, prev_n2 = parse_ffr_value(n1, n2, prev_n1, prev_n2, d)
                    else:
                        continue  # or raise error
                    row.append(f"{site} {formatted}")
            summary_per_site.append(row)

        layer_per_site.extend([layer] * site_len)

    return summary_per_site, layer_per_site


def get_holdout_layer_filter(layer_filter: list[str], results: dict[str, Any], w: str, c: str) -> list[str]:
    holdout_layer_filter = []
    for layer in layer_filter:
        if f"{w}#{layer}#{c}" not in results:
            continue
        holdout_layer_filter.append(layer)
    return holdout_layer_filter


def app() -> None:
    logger.debug("Loading Regression Result Viewer...")
    st.title("Regression Test Result Viewer")
    st.caption("Visualize Regression Test Result")

    r1_col1, r1_col2 = st.columns([1, 1])
    output_dir_default = "/mnt/fs0/MLE/ff_docker_output/mle_regression_test"
    with r1_col1:
        regression_result_parent_dir = st.text_input("Regression Result Directory", value=output_dir_default)
    with r1_col2:
        specific_dir = st.selectbox("Choose Date", os.listdir(regression_result_parent_dir))
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
        summary_p_site, layer_p_site = get_summary_p_site(layer_filter, st.session_state.normal, TABLE_COLUMNS, weeks)
        st.subheader("Regression Results Summary")
        st.dataframe(pd.DataFrame(summary, columns=headers, index=layer_filter))
        st.subheader("Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_p_site, columns=headers, index=layer_p_site))

    elif mode == "with Rule":
        if "rule" not in st.session_state:
            st.session_state.rule = {"regression_result": defaultdict(list), "ffr_check_memo": defaultdict()}
            for m in ["Normal", "with Rule"]:
                prefixes = [f"{w}_{lf}" for lf, w in itertools.product(layer_site_filter, rule_weeks)]
                result = api_helper.get_regression_result(regression_testcases, prefixes, regression_result_dir, m)
                st.session_state.rule["regression_result"].update(result["regression_result"])
                st.session_state.rule["ffr_check_memo"].update(result["ffr_check_memo"])
        regression_result = st.session_state.rule["regression_result"]
        logger.info(regression_result.keys())
        ffr_check_memo = st.session_state.rule["ffr_check_memo"]
        headers_weeks = [rule_weeks[0], f"{rule_weeks[0]}with Rule"]
        headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in headers_weeks]
        summary_p_site, layer_p_site = get_summary_p_site(layer_filter, regression_result, TABLE_COLUMNS, headers_weeks)
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
            st.session_state.r_only = api_helper.get_regression_result(
                regression_testcases, prefixes, regression_result_dir, mode
            )["regression_result"]
        headers_weeks = [f"{week}{mode}" for week in rule_weeks]
        summary_p_site, layer_p_site = get_summary_p_site(
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
        summary_per_site, layer_per_site = get_summary_p_site(
            holdout_layer_filter, st.session_state.holdout, RULE_TABLE_COLUMNS, weeks
        )
        st.subheader("Holdout Regression Results Summary (Per SITE):")
        st.dataframe(pd.DataFrame(summary_per_site, columns=site_headers, index=layer_per_site))

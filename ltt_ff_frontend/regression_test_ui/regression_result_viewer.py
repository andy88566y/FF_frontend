import itertools
import os
import re
import statistics
from collections import defaultdict
from typing import Any

import pandas as pd
import requests
import streamlit as st
from loguru import logger

from ltt_ff_frontend.constant import API_ROOT
from ltt_ff_frontend.regression_test_ui.regression_constant import LAYERS, LAYERS_FABS, RULE_LAYERS_FABS, RUN_MODES


CR_COL = "(CR OOS lots)"
FFR_COL = "(FFR OOS lots) Exclude flush"
AVG_FFR_COL = "avg ffr"
TABLE_COLUMNS = [CR_COL, FFR_COL]
RULE_TABLE_COLUMNS = [CR_COL, FFR_COL, AVG_FFR_COL]

ffr_check_memo: dict[str, Any] = defaultdict(list)


def save_false_filter_rates(test_case: str, resp: requests.Response) -> list:
    false_filter_rates = {
        stat["lot_id"]: stat["false_filter_rate"]
        for stat in resp.json()["filtered_stats"]
        if stat["false_filter_rate"] != -1
    }
    ffr_check_memo[test_case] = false_filter_rates
    return list(false_filter_rates.values())


def check_new_dir(result_dir: str) -> None:
    if "regression_result_dir" not in st.session_state or result_dir != st.session_state.regression_result_dir:
        logger.info("new result dir detected, clear old session_state value")
        for key in ["normal", "normal_p", "rule", "rule_warnings", "rule_only", "holdout", "holdout_p"]:
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


def get_table_summary(results: dict[str, list], weeks: list) -> list:
    table_summary = []
    for layer in LAYERS:
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

        table_summary.append(row)
    return table_summary


def get_table_summary_per_fab(results: dict[str, list], columns: list, weeks: list) -> tuple:
    table_summary_per_fab = []
    layer_per_fab = []

    for layer in LAYERS:
        fab_len = len(results[f"{weeks[0]}#{layer}#{columns[0]}"])
        for i in range(fab_len):
            row = []
            for col in columns:
                prev_n1 = prev_n2 = 9999
                for week in weeks:
                    fab, info = results[f"{week}#{layer}#{col}"][i].split()
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
                    row.append(f"{fab} {formatted}")
            table_summary_per_fab.append(row)

        layer_per_fab.extend([layer] * fab_len)

    return table_summary_per_fab, layer_per_fab


def get_regression_result(test_cases: dict[str, list], prefixes: list[str], result_dir: str, mode: str):
    regression_result = defaultdict(list)
    for prefix in prefixes:
        total_lot_count = 0
        ffr_total_lot_count = 0
        cr_oos_count = 0
        ffr_oos_count = 0
        ffr_non_critical_count = 0
        avg_ffr = -1.0
        if prefix not in test_cases:
            raise RuntimeError(
                f"No test results with prefix '{prefix}' were found. "
                "Please check if all the inference jobs were run and finished."
            )

        matched_test_cases = sorted(test_cases[prefix])
        for test_case in matched_test_cases:
            inference_result_dir = os.path.join(result_dir, test_case)
            r = requests.get(
                API_ROOT + "result/get_filtered_stats",
                json={"inference_result_dir": inference_result_dir, "mode": mode},
                timeout=15,
            )
            if r.ok:
                false_filter_rates = save_false_filter_rates(test_case, r)
                if false_filter_rates and mode != "Normal":
                    avg_ffr = statistics.mean(false_filter_rates)
                results = [filtered_stat["result"] for filtered_stat in r.json()["filtered_stats"]]

                count = len(results)
                total_lot_count += count

                cr_oos_count += sum(1 for r in results if r["Capture Rate"] == "OOS")
                if not re.match(r".*_flush$", test_case):
                    ffr_total_lot_count += count
                    ffr_oos_count += sum(1 for r in results if r["False Filter Rate"] == "OOS")
                    ffr_non_critical_count += sum(1 for r in results if r["False Filter Rate"] == "OOS, Non-Critical")
            else:
                st.error(f" ======== {inference_result_dir} ======== \n")
        critical_and_non_critical_count = ffr_oos_count + ffr_non_critical_count

        week, layer, fab = prefix.split("_")
        regression_result[f"{week}#{layer}#{CR_COL}"].append(f"{fab}: {cr_oos_count}/{total_lot_count}")
        regression_result[f"{week}#{layer}#{FFR_COL}"].append(
            f"{fab}: {ffr_oos_count}({critical_and_non_critical_count})/{ffr_total_lot_count}"
        )
        if mode != "Normal":
            regression_result[f"{week}#{layer}#{AVG_FFR_COL}"].append(f"{fab}: {avg_ffr}")
    return regression_result


def app() -> None:
    logger.debug("Loading Regression Result Viewer...")
    st.title("Regression Test Result Viewer")
    st.caption("Visualize Regression Test Result")
    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    output_dir_default = "/mnt/fs0/MLE/ff_docker_output/mle_regression_test"
    with r1_col1:
        regression_result_parent_dir = st.text_input("Regression Result Directory", value=output_dir_default)

        previous_week = st.text_input("Previous Week")
    with r1_col2:
        specific_dir = st.selectbox("Choose Date", os.listdir(regression_result_parent_dir))
        if regression_result_parent_dir and specific_dir:
            regression_result_dir = os.path.join(regression_result_parent_dir, specific_dir)
        check_new_dir(regression_result_dir)

        current_week = st.text_input("Current Week")
    mode = st.segmented_control("Run Mode", RUN_MODES, default="Normal")

    if mode is None:
        st.error("Run Mode Option can not be None!")
        return

    if previous_week and current_week and regression_result_dir:
        weeks = [f"W{previous_week}", f"W{current_week}"]
        rule_weeks = [f"W{current_week}", f"W{current_week}RULE"]
        r_only_weeks = [f"W{current_week}RULE"]
    else:
        return

    regression_test_cases = defaultdict(list)
    for _dir in os.listdir(regression_result_dir):
        key = "_".join(_dir.split("_")[:3])
        regression_test_cases[key].append(_dir)

    if mode == "Normal":
        if "normal" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(LAYERS_FABS, weeks)]
            regression_results = get_regression_result(
                regression_test_cases, prefixes, regression_result_dir, mode + "!"
            )

            headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in weeks]
            table_summary = get_table_summary(regression_results, weeks)
            st.session_state.normal = pd.DataFrame(table_summary, columns=headers, index=LAYERS)
            table_summary_per_fab, layer_per_fab = get_table_summary_per_fab(regression_results, TABLE_COLUMNS, weeks)
            st.session_state.normal_p = pd.DataFrame(table_summary_per_fab, columns=headers, index=layer_per_fab)

        st.subheader("Regression Results Summary")
        st.dataframe(st.session_state.normal)
        st.subheader("Regression Results Summary (Per FAB):")
        st.dataframe(st.session_state.normal_p)

    elif mode == "with Rule":
        if "rule" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(RULE_LAYERS_FABS, rule_weeks)]
            rule_results = get_regression_result(regression_test_cases, prefixes, regression_result_dir, mode)
            headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in rule_weeks]
            table_summary_per_fab, layer_per_fab = get_table_summary_per_fab(rule_results, TABLE_COLUMNS, rule_weeks)
            st.session_state.rule = pd.DataFrame(table_summary_per_fab, columns=headers, index=layer_per_fab)
            # list out ffr drop > 5% lots
            warnings = []

            for layer_fab in RULE_LAYERS_FABS:
                original_key = f"{rule_weeks[0]}_{layer_fab}"
                with_rule_key = f"{rule_weeks[1]}_{layer_fab}"
                for mask_type in ["flush", "production"]:
                    memo = ffr_check_memo[f"{with_rule_key}_{mask_type}"]
                    for lot_id, rule_ffr in memo.items():
                        no_rule_ffr = ffr_check_memo[f"{original_key}_{mask_type}"][lot_id]
                        if rule_ffr > no_rule_ffr:
                            warnings.append(
                                {
                                    "LayerFab": layer_fab,
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
                                    "LayerFab": layer_fab,
                                    "MaskType": mask_type,
                                    "LotID": lot_id,
                                    "NoRuleFFR": no_rule_ffr,
                                    "RuleFFR": rule_ffr,
                                    "Issue": "FFR decreased > 5% after applying RULE",
                                }
                            )
            st.session_state.rule_warnings = pd.DataFrame(warnings)

        st.subheader("RULE Regression Results Summary (Per FAB):")
        st.dataframe(st.session_state.rule)
        if not st.session_state.rule_warnings.empty:
            st.warning("⚠️ Some lots show unexpected FFR changes after applying RULE Model.")
            st.dataframe(st.session_state.rule_warnings)

    elif mode == "Rule only":
        if "rule_only" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(RULE_LAYERS_FABS, r_only_weeks)]
            r_only_results = get_regression_result(regression_test_cases, prefixes, regression_result_dir, mode)
            t_summary_per_fab, layer_per_fab = get_table_summary_per_fab(r_only_results, TABLE_COLUMNS, r_only_weeks)
            headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in r_only_weeks]
            st.session_state.rule_only = pd.DataFrame(t_summary_per_fab, columns=headers, index=layer_per_fab)
        st.subheader("RULE ONLY Regression Results Summary (Per FAB):")
        st.dataframe(st.session_state.rule_only)

    elif mode == "Holdout":
        holdout_test_cases = defaultdict(list)
        for _dir in os.listdir(regression_result_dir + "_holdout"):
            key = "_".join(_dir.split("_")[:3])
            holdout_test_cases[key].append(_dir)
        if "holdout" not in st.session_state:
            prefixes = [f"{w}_{lf}" for lf, w in itertools.product(LAYERS_FABS, weeks)]
            regression_results = get_regression_result(holdout_test_cases, prefixes, regression_result_dir, mode)

            headers = [f"{week} {col}" for col in TABLE_COLUMNS for week in weeks]
            table_summary = get_table_summary(regression_results, weeks)
            st.session_state.holdout = pd.DataFrame(table_summary, columns=headers, index=LAYERS)
            table_summary_per_fab, layer_per_fab = get_table_summary_per_fab(regression_results, TABLE_COLUMNS, weeks)
            st.session_state.holdout_p = pd.DataFrame(table_summary_per_fab, columns=headers, index=layer_per_fab)

        st.subheader("Holdout Regression Results Summary")
        st.dataframe(st.session_state.holdout)
        st.subheader("Holdout Regression Results Summary (Per FAB):")
        st.dataframe(st.session_state.holdout_p)

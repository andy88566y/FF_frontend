from typing import Any

import streamlit as st
from loguru import logger


CR_COL = "(CR OOS lots)"
FFR_COL = "(FFR OOS lots) Exclude flush"
AVG_FFR_COL = "avg ffr"
TABLE_COLUMNS = [CR_COL, FFR_COL]
RULE_TABLE_COLUMNS = [CR_COL, FFR_COL, AVG_FFR_COL]


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
                for info in results.get(f"{week}#{layer}#{col}", []):
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


def get_summary_per_site(layer_filter: list[str], results: dict[str, list], columns: list, weeks: list) -> tuple:
    summary_per_site = []
    layer_per_site = []

    for layer in layer_filter:
        site_len = len(results.get(f"{weeks[0]}#{layer}#{columns[0]}", []))
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
                    row.append(f"{site}  {formatted}")
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

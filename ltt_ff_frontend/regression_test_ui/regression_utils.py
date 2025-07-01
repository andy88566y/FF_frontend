import re
from typing import Any

import pandas as pd
import yaml
import copy

def gen_lots_stats(data_lots: dict[str, Any]) -> pd.DataFrame:
    lots_stats: dict[str, int] = {}
    for lot_info in data_lots.values():
        group_name = f"{lot_info['layer_group']}#{lot_info['site']}#{lot_info['mask_type']}"
        lots_stats[group_name] = lots_stats.get(group_name, 0) + 1
    formatted_stats = [(*gn.split("#"), cnt) for gn, cnt in sorted(lots_stats.items())]
    return (
        pd.DataFrame.from_records(formatted_stats, columns=["LG", "SITE", "mask_type", "cnt"])
        .pivot(index=["LG", "SITE"], columns="mask_type", values="cnt")
        .fillna(0)
        .astype("int32")
    )

def update_week(config_data: dict, new_week: str) -> None:
    for layer, weeks in config_data.items():
        sorted_week_keys = sorted(weeks.keys(), key=int)
        if str(new_week) in sorted_week_keys:
            return
        if len(sorted_week_keys) < 2:
            raise ValueError(f"⚠️ Layer '{layer}' has less than two weeks. Cannot perform update.")
        oldest_week = sorted_week_keys[0]
        second_oldest_week = sorted_week_keys[1]
        new_week_data = copy.deepcopy(weeks[second_oldest_week])

        del weeks[oldest_week]
        weeks[new_week] = new_week_data

def update_config_by_txt(config_data: dict, recipe_file: Any) -> None:
    raw_lines = recipe_file.getvalue().decode("utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    i = 0
    update_week(config_data,lines[0])

    while i < len(lines):
        week = int(lines[i])
        layer = lines[i + 1]
        site = lines[i + 2]
        i += 3
        recipes = []
        min_k = None
        top_k = None
        while i < len(lines) and not re.match(r"^\d+$", lines[i]):
            rs = re.match(r"(?P<model>\w+)\s+threshold:\s+(?P<th>[\d.]+)\s+threshold_c:\s+(?P<th_c>[\d.]+)", lines[i])
            min_k_match = re.match(r"min_[k]:\s*(\d+)", lines[i])
            top_k_match = re.match(r"top_[k]:\s*(\d+)", lines[i])
            if rs:
                recipe_id = rs.group("model")
                threshold = float(rs.group("th"))
                threshold_c = float(rs.group("th_c"))
                recipes.append([recipe_id, threshold, threshold_c])
            elif min_k_match:
                min_k = int(min_k_match.group(1))
            elif top_k_match:
                top_k = int(top_k_match.group(1))
            else:
                raise ValueError(f"Could not parse recipe line: {lines[i]}")
            i += 1
        config_data.setdefault(layer, {}).setdefault(str(week), {})[site] = {
            "recipes": recipes,
        }
        if min_k:
            config_data[layer][week][site]["min_k"] = min_k
        if top_k:
            config_data[layer][week][site]["top_k"] = top_k


def update_config_by_yaml(config_data: dict, recipe_yaml: Any) -> None:
    file_name = recipe_yaml.name.split(".")[0]
    update_data = yaml.safe_load(recipe_yaml)
    match = re.match(r"^(W\d+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)$", file_name)
    if not match:
        raise ValueError(f"Invalid filename format: {file_name}. Expected format: WX_layer_site")
    week, layer, site = match.groups()
    update_week(config_data, week[1:])
    update_data["recipes"] = [tuple(model.values()) for model in update_data["recipes"]]

    config_data.setdefault(layer, {}).setdefault(week, {})[site] = update_data

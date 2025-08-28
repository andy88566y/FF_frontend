import copy
import os
import re
from typing import Any

import pandas as pd
import yaml

from ltt_ff_frontend.helpers import api_helper


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


def get_detailed_stats(stats: list[Any]) -> pd.DataFrame:
    stats_df = (
        pd.DataFrame.from_records(data=stats)
        .pivot(
            index=["layer_group", "site", "tool", "pixel_size", "mask_type", "lrf_type", "lot_id"],
            columns=["is_defect", "class_type"],
            values="count",
        )
        .fillna(0)
        .astype("int32")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return stats_df


class Regression_Recipe:
    def __init__(self, recipe_config: dict[str, Any]):
        self.recipe = recipe_config
        self.is_valid_recipe()
        self.recipe_model_mapping()

    def get_layer_group(self) -> list[str]:
        return sorted(self.recipe.keys())

    def get_week(self) -> list[str]:
        return sorted(list(self.recipe.values())[0].keys())

    def get_site(self) -> list[str]:
        site_set = set()
        for week_site_recipe in self.recipe.values():
            for site_recipe in week_site_recipe.values():
                site_set |= set(site_recipe.keys())
        return sorted(site_set)

    def update_week(self, new_week: str) -> None:
        for layer, weeks in self.recipe.items():
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

    def update_by_txt(self, recipe_file: Any) -> None:
        if isinstance(recipe_file, str):
            raw_lines = recipe_file.splitlines()
        else:
            raw_lines = recipe_file.getvalue().decode("utf-8").splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]
        week = lines[0].strip()
        self.update_week(week)
        current_layer = None

        site_pattern = re.compile(r"\[(.*?)\]")
        recipe_pattern = re.compile(r"(\d{4})-([a-fA-F0-9]+)")
        threshold_pattern = re.compile(r"\(TH:(\d*\.?\d*)(?:,THC:(\d*\.?\d*))?\)")
        min_k_pattern = re.compile(r"min_k\s*=\s*(\d+)")
        top_k_pattern = re.compile(r"top_k\s*=\s*(\d+)")

        for line in lines[1:]:
            if not line:
                continue
            if line.isupper():
                current_layer = line
                if current_layer == "ODPO":
                    current_layer = "OD"
                continue

            filter_line = line
            for old, new in [("*new*", ""), ("TH: ", "TH:"), (" THC: ", "THC:"), (" \u200bTHC: ", "THC:")]:
                filter_line = filter_line.replace(old, new)

            site = None
            recipe: list[dict] = []
            min_k = None
            top_k = None
            model_id = None

            for item in filter_line.split():
                site_matches = site_pattern.findall(item)
                if site_matches:
                    if len(site_matches) == 2:
                        current_layer, site = site_matches
                    else:
                        site = site_matches[0]
                    continue

                if recipe_pattern.match(item):
                    _, model_id = item.split("-")
                    continue

                if item.startswith("RULE"):
                    model_id = item
                    continue

                th_match = threshold_pattern.search(item)
                if th_match:
                    th = float(th_match.group(1))
                    th_c = float(th_match.group(2)) if th_match.group(2) else None

                    if not model_id:
                        raise ValueError("threshold should come after model id!")

                    setting = {"model_id": model_id, "threshold": th}
                    if th_c is not None:
                        setting["threshold_c"] = th_c
                    recipe.append(setting)

                    continue

                min_k_match = min_k_pattern.search(item)
                if min_k_match:
                    min_k = int(min_k_match.group(1))
                    continue

                top_k_match = top_k_pattern.search(item)
                if top_k_match:
                    top_k = int(top_k_match.group(1))
                    continue

                raise ValueError(f"Cannot parse line {line}, item: {item}")

            self.recipe[current_layer][week][site] = {"recipes": recipe}
            if min_k is not None:
                self.recipe[current_layer][week][site]["min_k"] = min_k
            if top_k is not None:
                self.recipe[current_layer][week][site]["top_k"] = top_k

    def update_by_yaml(self, recipe_yaml: Any) -> None:
        file_name = recipe_yaml.name.split(".")[0]
        update_data = yaml.safe_load(recipe_yaml)
        match = re.match(r"^(W\d+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)$", file_name)
        if not match:
            raise ValueError(f"Invalid filename format: {file_name}. Expected format: WX_layer_site")
        week, layer, site = match.groups()
        week = week[1:]
        self.update_week(week)
        self.recipe.setdefault(layer, {}).setdefault(week, {})[site] = update_data

    def recipe_model_mapping(self) -> None:
        model_list = api_helper.get_base_models()
        models_mapping = {
            m.split("#", maxsplit=-1)[-1].split(".", maxsplit=1)[0]: m for m in model_list if m.startswith("base")
        }
        for week_dict in self.recipe.values():
            for site_config in week_dict.values():
                for recipes in site_config.values():
                    for r in recipes["recipes"]:
                        mid = r["model_id"]
                        is_valid_model = mid.endswith(".pth") and mid.startswith("base") or mid.startswith("RULE")
                        model_name = mid if is_valid_model else models_mapping.get(mid)
                        if model_name is None:
                            raise ValueError(f"Recipe contains invalid model name: {mid} !")
                        r["model_name"] = model_name
                        del r["model_id"]

    def make_recipe_dir(self, output_dir: str) -> None:
        recipe_root = os.path.join(output_dir, "recipes")
        os.makedirs(recipe_root, exist_ok=True)
        for layer_group, week_dict in self.recipe.items():
            layer_root = os.path.join(recipe_root, layer_group)
            os.makedirs(layer_root, exist_ok=True)
            for week, site_config in week_dict.items():
                week_root = os.path.join(layer_root, f"W{week}")
                os.makedirs(week_root, exist_ok=True)
                for site, recipes in site_config.items():
                    recipe_path = os.path.join(week_root, f"W{week}_{layer_group}_{site}.yaml")
                    with open(recipe_path, "w") as f:
                        yaml.dump(recipes, f, sort_keys=True)

    def is_valid_recipe(self) -> None:
        if not isinstance(self.recipe, dict):
            raise ValueError("Top-level config must be a dictionary.")

        for layer_group, week_data in self.recipe.items():
            if not isinstance(week_data, dict):
                raise ValueError(f"LayerGroup '{layer_group}' must map to a dictionary of weeks.")

            for week, site_data in week_data.items():
                if not isinstance(site_data, dict):
                    raise ValueError(
                        f"Week '{week}' under LayerGroup '{layer_group}' must map to a dictionary of sites."
                    )

                for site, recipe_block in site_data.items():
                    if not isinstance(recipe_block, dict) or "recipes" not in recipe_block:
                        raise ValueError(f"Site '{site}' under Week '{week}' must contain a 'recipes' list.")

                    recipes = recipe_block["recipes"]
                    if not isinstance(recipes, list):
                        raise ValueError(f"'recipes' under Site '{site}' must be a list.")

                    for i, recipe in enumerate(recipes):
                        if not isinstance(recipe, dict):
                            raise ValueError(f"Recipe #{i} under Site '{site}' must be a dictionary.")
                        if "model_id" not in recipe:
                            raise ValueError(f"Recipe #{i} under Site '{site}' is missing 'model_id'.")
                        if "threshold" not in recipe:
                            raise ValueError(f"Recipe #{i} under Site '{site}' is missing 'threshold'.")
                        # Optional: threshold_c
                        if "threshold_c" in recipe and not isinstance(recipe["threshold_c"], (int, float)):
                            raise ValueError(f"'threshold_c' in Recipe #{i} under Site '{site}' must be a number.")

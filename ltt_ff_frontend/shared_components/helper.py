from typing import Any
from ltt_ff_frontend.constant import BLANK_MODEL

# put only codes like: format strings, aggregate data
# for generating figure, extract to one seperate component file

def aggregate_lists(
    raw_data: tuple[list[list[int]], list[list[float]], list[list[int]]],
    meta_list: list[dict[str, Any]]
) -> tuple[list[int], list[float], list[int], list[str]]:
    defect_id_lists, prob_lists, ans_lists = raw_data
    lot_id_lists = [meta["lot_id"] for meta in meta_list]
    aggregate_id_list, aggregate_prob_list, aggregate_ans_list, aggregate_lot_id_list = [], [], [], []
    for defect_id_list, prob_list, ans_list, lot_id in zip(defect_id_lists, prob_lists, ans_lists, lot_id_lists):
        aggregate_id_list.extend(defect_id_list)
        aggregate_prob_list.extend(prob_list)
        aggregate_ans_list.extend(ans_list)
        aggregate_lot_id_list.extend([lot_id] * len(defect_id_list))

    return (aggregate_id_list, aggregate_prob_list, aggregate_ans_list, aggregate_lot_id_list)

def format_model_name(name: str | None) -> str:
    if name is None:
        return "SCRATCH"
    if name == BLANK_MODEL:
        return BLANK_MODEL
    # For base model name (e.g. base/LTT_SW#x9u#N3#M0-M2#20250124T000000Z#55032dae#55032dae.encrypted.pth)
    if "/" in name:
        model_paths = name.split("/")
        model_type = model_paths[-2]
        model_name = model_paths[-1]
        return f"[{model_type}] {model_name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"
    # For output model name (e.g. 13feb_minye#x9u#tl#lg20250213T151435Z#55032dae#5fb1017f)
    else:
        return f"{name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"
from typing import Any

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
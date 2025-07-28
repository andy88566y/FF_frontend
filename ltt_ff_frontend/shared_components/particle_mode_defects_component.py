from typing import Any

import pandas as pd
import streamlit as st


def gen(particle_mode_info: dict[str, Any]) -> None:
    df = pd.DataFrame(
        columns=[
            "Lot ID",
            "Particle-mode-only defects",
            "Relaxed particle mode defects",
            "U/L particle mode defects",
            "LRF Path",
        ]
    )

    all_particle_mode_defect_lists = particle_mode_info["all_particle_mode_defect_lists"]
    particle_mode_only_defect_lists = all_particle_mode_defect_lists["particle_mode_only_defect_lists"]
    relaxed_particle_mode_lists = all_particle_mode_defect_lists["relaxed_particle_mode_lists"]
    ul_particle_mode_lists = all_particle_mode_defect_lists["ul_particle_mode_lists"]

    lrf_paths = particle_mode_info["lrf_path_list"]

    df_dict = {}

    for particle_mode_defects_per_lot, lrf_path in zip(particle_mode_only_defect_lists, lrf_paths):
        for k, v in particle_mode_defects_per_lot.items():
            df_dict[k] = {"Particle-mode-only defects": v, "LRF Path": lrf_path}

    for relaxed_particle_mode_defects_per_lot, lrf_path in zip(relaxed_particle_mode_lists, lrf_paths):
        for k, v in relaxed_particle_mode_defects_per_lot.items():
            df_dict[k].update({"Relaxed particle mode defects": v, "LRF Path": lrf_path})

    for ul_particle_mode_defects_per_lot, lrf_path in zip(ul_particle_mode_lists, lrf_paths):
        for k, v in ul_particle_mode_defects_per_lot.items():
            df_dict[k].update({"U/L particle mode defects": v, "LRF Path": lrf_path})

    # Update df_dict to conform to steamlit requirements
    for lot_id, all_lists in df_dict.items():
        df.loc[len(df)] = pd.Series(
            {
                "Lot ID": lot_id,
                "Particle-mode-only defects": all_lists.get("Particle-mode-only defects", ["None"]),
                "Relaxed particle mode defects": all_lists.get("Relaxed particle mode defects", ["None"]),
                "U/L particle mode defects": all_lists.get("U/L particle mode defects", ["None"]),
                "LRF Path": lrf_path,
            }
        )

    st.dataframe(df)

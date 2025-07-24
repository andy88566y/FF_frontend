from typing import Any

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper


def gen(particle_mode_info: dict[str, Any]) -> None:
    df = pd.DataFrame(columns=["Lot ID", "Particle-mode-only defects", "LRF Path"])

    particle_mode_lists = particle_mode_info["particle_mode_only_defect_list"]
    lrf_paths = particle_mode_info["lrf_path_list"]

    for particle_mode_defects_per_lot, lrf_path in zip(particle_mode_lists, lrf_paths):
        for k, v in particle_mode_defects_per_lot.items():
            df.loc[len(df)] = pd.Series({"Lot ID": k, "Particle-mode-only defects": v, "LRF Path": lrf_path})
    st.dataframe(df)

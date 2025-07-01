from typing import Any

import pandas as pd
import streamlit as st

from ltt_ff_frontend.helpers import api_helper


def gen(particle_mode_only_defects: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(columns=["Lot ID", "Particle-mode-only defects"])
    for missed_defect_per_lot in particle_mode_only_defects:
        for k, v in missed_defect_per_lot.items():
            df.loc[len(df)] = pd.Series(
                {"Lot ID": k, "Particle-mode-only defects": v if len(v) > 0 else ["No particleModeOnly defects!"]}
            )
    st.dataframe(df)

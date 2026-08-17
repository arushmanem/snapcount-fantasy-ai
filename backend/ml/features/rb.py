from __future__ import annotations
import pandas as pd
from .base import mean_all, mean_last_n, std_all, count_games

RB_FEATURES = [
    "fp_avg",
    "fp_avg_last_1",
    "fp_avg_last_3",
    "fp_avg_last_5",
    "fp_std",
    "rush_att_avg_last_3",
    "rush_yards_avg_last_3",
    "rush_tds_avg_last_3",
    "targets_avg_last_3",
    "receptions_avg_last_3",
    "rec_yards_avg_last_3",
    "snap_pct_avg_last_3",
    "games_played",
    # matchup/environment/defense
    "is_home",
    "temp_f",
    "wind_mph",
    "is_dome",
    "is_outdoor",
    "opp_fp_allowed_vs_pos",
]

def build_rb_features(
    history: pd.DataFrame,
    fp_col: str,
    *,
    is_home: int,
    temp_f: float,
    wind_mph: float,
    is_dome: int,
    is_outdoor: int,
    opp_fp_allowed_vs_pos: float,
) -> dict:
    return {
        "fp_avg": mean_all(history, fp_col),
        "fp_avg_last_1": mean_last_n(history, fp_col, 1),
        "fp_avg_last_3": mean_last_n(history, fp_col, 3),
        "fp_avg_last_5": mean_last_n(history, fp_col, 5),
        "fp_std": std_all(history, fp_col),

        "rush_att_avg_last_3": mean_last_n(history, "rushing_attempts", 3),
        "rush_yards_avg_last_3": mean_last_n(history, "rushing_yards", 3),
        "rush_tds_avg_last_3": mean_last_n(history, "rushing_tds", 3),

        "targets_avg_last_3": mean_last_n(history, "targets", 3),
        "receptions_avg_last_3": mean_last_n(history, "receptions", 3),
        "rec_yards_avg_last_3": mean_last_n(history, "receiving_yards", 3),

        "snap_pct_avg_last_3": mean_last_n(history, "snap_pct", 3),
        "games_played": count_games(history),

        "is_home": float(is_home),
        "temp_f": float(temp_f),
        "wind_mph": float(wind_mph),
        "is_dome": float(is_dome),
        "is_outdoor": float(is_outdoor),
        "opp_fp_allowed_vs_pos": float(opp_fp_allowed_vs_pos),
    }

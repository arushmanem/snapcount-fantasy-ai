from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

from ml.features.qb import build_qb_features, QB_FEATURES
from ml.features.rb import build_rb_features, RB_FEATURES
from ml.features.wr import build_wr_features, WR_FEATURES
from ml.features.te import build_te_features, TE_FEATURES

POS_BUILDERS = {
    "QB": (build_qb_features, QB_FEATURES),
    "RB": (build_rb_features, RB_FEATURES),
    "WR": (build_wr_features, WR_FEATURES),
    "TE": (build_te_features, TE_FEATURES),
}

FP_COL_BY_FORMAT = {
    "ppr": "fantasy_points_ppr",
    "half_ppr": "fantasy_points_half_ppr",
    "standard": "fantasy_points",
}


def schedule_reg_only(df: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in df.columns:
        return df[df["season_type"] == "REG"].copy()
    if "game_type" in df.columns:
        return df[df["game_type"] == "REG"].copy()
    return df.copy()


def _get_first_existing(row: pd.Series, candidates: list[str], default=None):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


def extract_weather_flags(row: pd.Series) -> tuple[float, float, int, int]:
    """
    Best-effort extraction across varying nfl_data_py schedule schemas.
    If unknown, return zeros.
    """
    temp = _get_first_existing(row, ["temp", "temperature", "weather_temp", "game_temperature"], 0.0)
    wind = _get_first_existing(row, ["wind", "wind_mph", "weather_wind", "weather_wind_mph"], 0.0)

    roof = _get_first_existing(row, ["roof", "stadium_roof", "roof_type"], "")
    roof_str = str(roof).lower() if roof is not None else ""

    # crude but effective
    is_dome = 1 if any(k in roof_str for k in ["dome", "closed", "indoor"]) else 0
    is_outdoor = 1 if any(k in roof_str for k in ["outdoor", "open"]) else (0 if is_dome else 0)

    # sometimes there is an explicit boolean/flag for dome
    dome_flag = _get_first_existing(row, ["is_dome", "dome"], None)
    if dome_flag is not None and str(dome_flag).lower() in ["1", "true", "yes"]:
        is_dome = 1

    try:
        temp_f = float(temp) if temp is not None else 0.0
    except Exception:
        temp_f = 0.0

    try:
        wind_mph = float(wind) if wind is not None else 0.0
    except Exception:
        wind_mph = 0.0

    return temp_f, wind_mph, int(is_dome), int(is_outdoor)


def build_schedule_team_map(df_sched_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Creates mapping rows for BOTH home and away team:
    (season, week, team) -> opponent, is_home, temp_f, wind_mph, is_dome, is_outdoor
    """
    needed = ["season", "week", "home_team", "away_team"]
    for c in needed:
        if c not in df_sched_reg.columns:
            raise ValueError(f"Schedule missing required column: {c}")

    rows = []
    for _, r in df_sched_reg.iterrows():
        season = int(r["season"])
        week = int(r["week"])
        home = str(r["home_team"])
        away = str(r["away_team"])
        temp_f, wind_mph, is_dome, is_outdoor = extract_weather_flags(r)

        # home team view
        rows.append({
            "season": season,
            "week": week,
            "team": home,
            "opponent": away,
            "is_home": 1,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "is_dome": is_dome,
            "is_outdoor": is_outdoor,
        })
        # away team view
        rows.append({
            "season": season,
            "week": week,
            "team": away,
            "opponent": home,
            "is_home": 0,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "is_dome": is_dome,
            "is_outdoor": is_outdoor,
        })

    return pd.DataFrame(rows)


def build_defense_allowed_table(
    df_stats: pd.DataFrame,
    sched_map: pd.DataFrame,
    fp_col: str,
) -> pd.DataFrame:
    """
    Returns df with columns:
      season, week, defense_team, position, opp_fp_allowed_vs_pos

    Uses ONLY information from weeks < current week by shifting expanding mean by 1 week.
    """
    # attach opponent (defense) to each offensive player-week row
    # offensive team is recent_team
    base = df_stats.dropna(subset=["season", "week", "position", "recent_team", fp_col]).copy()
    base["season"] = base["season"].astype(int)
    base["week"] = base["week"].astype(int)
    base["team"] = base["recent_team"].astype(str)

    merged = base.merge(
        sched_map[["season", "week", "team", "opponent"]],
        on=["season", "week", "team"],
        how="inner",
    )
    merged["defense_team"] = merged["opponent"].astype(str)
    merged["fp"] = merged[fp_col].astype(float)

    # weekly points allowed by defense vs position
    weekly_allowed = (
        merged.groupby(["season", "week", "defense_team", "position"], as_index=False)["fp"]
        .mean()
        .rename(columns={"fp": "allowed_this_week"})
        .sort_values(["season", "defense_team", "position", "week"])
    )

    # expanding mean through prior week (shifted)
    def _expanding_prior(s: pd.Series) -> pd.Series:
        return s.expanding().mean().shift(1)

    weekly_allowed["opp_fp_allowed_vs_pos"] = (
        weekly_allowed
        .groupby(["season", "defense_team", "position"])["allowed_this_week"]
        .transform(_expanding_prior)
    )

    return weekly_allowed[["season", "week", "defense_team", "position", "opp_fp_allowed_vs_pos"]]


def build_position_dataset(
    df_stats: pd.DataFrame,
    sched_map: pd.DataFrame,
    def_allowed: pd.DataFrame,
    position: str,
    fp_col: str,
) -> pd.DataFrame:
    df_pos = df_stats[df_stats["position"] == position].copy()
    if df_pos.empty:
        return pd.DataFrame()

    # normalize player display name
    if "player_display_name" not in df_pos.columns and "player_name" in df_pos.columns:
        df_pos["player_display_name"] = df_pos["player_name"]

    df_pos = df_pos.dropna(subset=[fp_col, "week", "season", "recent_team", "player_display_name"])
    df_pos["season"] = df_pos["season"].astype(int)
    df_pos["week"] = df_pos["week"].astype(int)
    df_pos["team"] = df_pos["recent_team"].astype(str)

    # attach matchup meta for the label week t
    df_pos = df_pos.merge(
        sched_map,
        on=["season", "week", "team"],
        how="inner",
    )

    builder, feature_cols = POS_BUILDERS[position]

    # for fallback defense signal when missing early-season
    league_allowed = (
        def_allowed.dropna(subset=["opp_fp_allowed_vs_pos"])
        .groupby(["season", "week", "position"])["opp_fp_allowed_vs_pos"]
        .mean()
        .reset_index()
        .rename(columns={"opp_fp_allowed_vs_pos": "league_opp_fp_allowed_vs_pos"})
    )

    rows: list[dict] = []
    # group by player-season: no cross-season leakage
    for (season, player), g in df_pos.groupby(["season", "player_display_name"], sort=False):
        g = g.sort_values("week")

        # map week -> row dict (label+meta)
        week_to_row = {int(w): r for w, r in zip(g["week"].astype(int), g.to_dict(orient="records"))}
        weeks = sorted(week_to_row.keys())

        for t in weeks:
            if t < 2:
                continue  # skip week 1 for v0

            cutoff = t - 1

            # history uses ONLY <= cutoff
            hist = g[g["week"] <= cutoff].copy()
            if len(hist) == 0:
                continue

            label_row = week_to_row[t]
            y = float(label_row.get(fp_col, 0.0))
            if pd.isna(y):
                continue

            opponent = str(label_row["opponent"])
            is_home = int(label_row["is_home"])
            temp_f = float(label_row.get("temp_f", 0.0) or 0.0)
            wind_mph = float(label_row.get("wind_mph", 0.0) or 0.0)
            is_dome = int(label_row.get("is_dome", 0) or 0)
            is_outdoor = int(label_row.get("is_outdoor", 0) or 0)

            # defense allowed lookup: use week=t row which already means "through week t-1" due to shift
            da = def_allowed[
                (def_allowed["season"] == int(season))
                & (def_allowed["week"] == int(t))
                & (def_allowed["defense_team"] == opponent)
                & (def_allowed["position"] == position)
            ]
            if len(da) > 0 and pd.notna(da.iloc[0]["opp_fp_allowed_vs_pos"]):
                opp_fp_allowed = float(da.iloc[0]["opp_fp_allowed_vs_pos"])
            else:
                # fallback to league average for that season/week/position (also shifted)
                la = league_allowed[
                    (league_allowed["season"] == int(season))
                    & (league_allowed["week"] == int(t))
                    & (league_allowed["position"] == position)
                ]
                opp_fp_allowed = float(la.iloc[0]["league_opp_fp_allowed_vs_pos"]) if len(la) else 0.0

            feat = builder(
                hist,
                fp_col,
                is_home=is_home,
                temp_f=temp_f,
                wind_mph=wind_mph,
                is_dome=is_dome,
                is_outdoor=is_outdoor,
                opp_fp_allowed_vs_pos=opp_fp_allowed,
            )

            row = {
                "season": int(season),
                "week": int(t),
                "cutoff_week": int(cutoff),
                "player": player,
                "position": position,
                "team": str(hist.sort_values("week")["team"].iloc[-1]),
                "opponent": opponent,
                "is_home": int(is_home),
                "temp_f": float(temp_f),
                "wind_mph": float(wind_mph),
                "is_dome": int(is_dome),
                "is_outdoor": int(is_outdoor),
                "opp_fp_allowed_vs_pos": float(opp_fp_allowed),
                "y": y,
            }

            for c in feature_cols:
                row[c] = float(feat.get(c, 0.0))

            rows.append(row)

    return pd.DataFrame(rows)


def split_by_season(df: pd.DataFrame, train_max_season: int, val_season: int, test_season: int):
    train = df[df["season"] <= train_max_season].copy()
    val = df[df["season"] == val_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--formats", nargs="+", type=str, default=["ppr", "half_ppr", "standard"])
    ap.add_argument("--train_max_season", type=int, default=2023)
    ap.add_argument("--val_season", type=int, default=2024)
    ap.add_argument("--test_season", type=int, default=2025)
    args = ap.parse_args()

    print(f"Loading schedules for seasons: {args.seasons}")
    df_sched = nfl.import_schedules(args.seasons)
    df_sched_reg = schedule_reg_only(df_sched)
    sched_map = build_schedule_team_map(df_sched_reg)

    out_datasets = Path("artifacts/datasets")
    out_schemas = Path("artifacts/schemas")
    out_datasets.mkdir(parents=True, exist_ok=True)
    out_schemas.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        fp_col = FP_COL_BY_FORMAT[fmt]
        print(f"\n==============================")
        print(f"Building datasets for format: {fmt} ({fp_col})")
        print(f"==============================")

        print(f"Loading weekly data for seasons: {args.seasons}")
        df_stats = nfl.import_weekly_data(args.seasons)

        # defense allowed table is format-specific (depends on fp_col)
        print("Building defense allowed table (no leakage, shifted expanding mean)...")
        def_allowed = build_defense_allowed_table(df_stats, sched_map, fp_col)

        for pos in ["QB", "RB", "WR", "TE"]:
            print(f"\nBuilding dataset for {pos} - {fmt}...")
            df_pos = build_position_dataset(df_stats, sched_map, def_allowed, pos, fp_col)
            if df_pos.empty:
                print(f"  No rows for {pos} ({fmt}) (skipping).")
                continue

            builder, feature_cols = POS_BUILDERS[pos]

            schema_path = out_schemas / f"{pos}_{fmt}_features.json"
            with open(schema_path, "w") as f:
                json.dump(feature_cols, f, indent=2)
            print(f"  Wrote schema: {schema_path}")

            train, val, test = split_by_season(df_pos, args.train_max_season, args.val_season, args.test_season)

            train_path = out_datasets / f"{pos}_{fmt}_train.parquet"
            val_path = out_datasets / f"{pos}_{fmt}_val.parquet"
            test_path = out_datasets / f"{pos}_{fmt}_test.parquet"

            train.to_parquet(train_path, index=False)
            val.to_parquet(val_path, index=False)
            test.to_parquet(test_path, index=False)

            print(f"  Rows: total={len(df_pos)} train={len(train)} val={len(val)} test={len(test)}")
            print(f"  Saved: {train_path}")
            print(f"  Saved: {val_path}")
            print(f"  Saved: {test_path}")

    print("\nDone building ALL datasets/schemas.")


if __name__ == "__main__":
    main()

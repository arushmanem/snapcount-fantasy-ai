from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nfl_data_py as nfl
import pandas as pd
import os
from typing import List, Optional, Tuple

# Gemini (new SDK)
import google.genai as genai

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Config
# ----------------------------
CURRENT_YEAR = int(os.environ.get("NFL_SEASON", "2024"))

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# Use a model your key can actually see (e.g. models/gemini-2.5-flash)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

if not GOOGLE_API_KEY:
    print("❌ ERROR: API Key NOT found! Check your .env file.")
else:
    print(f"✅ API Key found: {GOOGLE_API_KEY[:5]}...")

client: Optional[genai.Client] = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"api_version": "v1"})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data loading
# ----------------------------
def load_data(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_stats = nfl.import_weekly_data([season])
    df_schedule = nfl.import_schedules([season])

    if "player_name" in df_stats.columns and "player_display_name" not in df_stats.columns:
        df_stats["player_display_name"] = df_stats["player_name"]

    return df_stats, df_schedule


df_stats, df_schedule = load_data(CURRENT_YEAR)


def schedule_reg_only(df: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in df.columns:
        return df[df["season_type"] == "REG"].copy()
    if "game_type" in df.columns:
        return df[df["game_type"] == "REG"].copy()
    return df.copy()


DF_SCHED_REG = schedule_reg_only(df_schedule)

AVAILABLE_WEEKS: List[int] = sorted(
    [int(w) for w in DF_SCHED_REG["week"].dropna().unique().tolist()]
)
DEFAULT_WEEK: int = max(AVAILABLE_WEEKS) if AVAILABLE_WEEKS else 18

# ----------------------------
# Helpers
# ----------------------------
COL_MAP = {
    "ppr": "fantasy_points_ppr",
    "half_ppr": "fantasy_points_half_ppr",
    "standard": "fantasy_points",
}


def get_player_data(name: str) -> pd.DataFrame:
    if df_stats.empty:
        return pd.DataFrame()
    return df_stats[df_stats["player_display_name"] == name].copy()


def clamp_week(week: int) -> int:
    if not AVAILABLE_WEEKS:
        return max(1, week)
    if week < AVAILABLE_WEEKS[0]:
        return AVAILABLE_WEEKS[0]
    if week > AVAILABLE_WEEKS[-1]:
        return AVAILABLE_WEEKS[-1]
    return week


def get_opponent_for_team_week(team: str, week: int) -> tuple[str, bool]:
    """
    Returns (opponent_team, is_home). If no game that week => ("BYE", False).
    """
    week = clamp_week(int(week))
    games = DF_SCHED_REG[
        (DF_SCHED_REG["week"] == week)
        & ((DF_SCHED_REG["home_team"] == team) | (DF_SCHED_REG["away_team"] == team))
    ]
    if games.empty:
        return ("BYE", False)

    row = games.iloc[0]
    if row["home_team"] == team:
        return (str(row["away_team"]), True)
    return (str(row["home_team"]), False)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _recent_form_line(p: dict, stats_context: str) -> str:
    games_played = int(p.get("games_played", 0))
    if games_played == 0:
        return f"- Recent form: No games played {stats_context}"
    n = int(p.get("recent_n", 1))
    avg = p.get("recent_avg", 0.0)
    return f"- Recent form: {avg} avg over last {n} game(s) {stats_context}"


def build_prompt(winner: dict, loser: dict, scoring: str, matchup_week: int, cutoff_week: int) -> str:
    """
    matchup_week = the week you're deciding for (opponent/BYE)
    cutoff_week  = the last week of data you're allowed to use (matchup_week - 1)
    """
    if cutoff_week == 0:
        stats_context = "(no prior games yet)"
    else:
        stats_context = f"(through Week {cutoff_week})"

    bye_note = ""
    if winner.get("opponent") == "BYE" or loser.get("opponent") == "BYE":
        bye_note = (
            "\nImportant: If a player is on BYE, say they are on a BYE and that they cannot be started.\n"
        )

    return f"""You are a decisive fantasy football analyst.
    Matchup Week: {matchup_week}
    Scoring: {scoring.upper()}{bye_note}

    Player A (recommended START):
    - Name: {winner['name']}
    - Projected: {winner['projected_points']}
    - Season Avg {stats_context}: {winner['avg_points']}
    {_recent_form_line(winner, stats_context)}
    - Opponent (Week {matchup_week}): {winner['opponent']}
    - Home: {winner['is_home']}

    Player B (SIT):
    - Name: {loser['name']}
    - Projected: {loser['projected_points']}
    - Season Avg {stats_context}: {loser['avg_points']}
    {_recent_form_line(loser, stats_context)}
    - Opponent (Week {matchup_week}): {loser['opponent']}
    - Home: {loser['is_home']}

    Write roughly 3-5 sentences:
    1) Start recommendation (one sentence, confident).
    2) Use the opponent/BYE + the stats context above (be specific).
    3) Why Player B is the weaker play this week (no hedging, no clichés).

    Speak reasonably. If it's a close call, be honest that it's a close call and explain why
    you chose the person you chose. Show compotent reasoning and explain how close of a call or how much of an undeniable start it is.
    Confident but reasonable is the main motto to follow. Don't make extreme statements like it's not close unless the projections are like 4 points apart or something like that. 
    """


def generate_ai_analysis(winner: dict, loser: dict, scoring: str, matchup_week: int, cutoff_week: int) -> str:
    fallback = (
        f"Start {winner['name']} in Week {matchup_week}. "
        f"They project {winner['projected_points']:.1f} vs {loser['projected_points']:.1f} for {loser['name']} "
        f"({scoring.upper()}). "
        f"The numbers give {winner['name']} the better floor this week."
    )

    if client is None:
        return fallback

    prompt = build_prompt(winner, loser, scoring, matchup_week, cutoff_week)

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (resp.text or "").strip()
        return text if text else fallback
    except Exception as e:
        print(f"AI Error: {e}")
        return fallback


def get_radar_data(
    p1_name: str,
    p2_name: str,
    position: str,
    matchup_week: int | None,
    format: str,
    ) -> List[dict]:
    """
    Radar percentiles computed using data through (matchup_week - 1).
    """
    categories = {
        "QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds"],
        "RB": ["rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards"],
        "WR": ["targets", "receptions", "receiving_yards", "receiving_tds", "rushing_yards"],
        "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", "fantasy_points_scoring"],
    }

    feats = categories.get(position, ["fantasy_points_scoring"])
    target_col = COL_MAP.get(format, "fantasy_points_ppr")

    df_pos = df_stats[df_stats["position"] == position].copy()
    if df_pos.empty:
        return [{"subject": f, "A": 0, "B": 0, "fullMark": 100} for f in feats]

    # ✅ Use cutoff = matchup_week - 1 for radar
    if matchup_week is not None:
        wk = clamp_week(int(matchup_week))
        cutoff = max(0, wk - 1)
        if cutoff > 0:
            df_pos = df_pos[df_pos["week"] <= cutoff]
        else:
            df_pos = df_pos.iloc[0:0]

    if df_pos.empty:
        return [{"subject": f, "A": 0, "B": 0, "fullMark": 100} for f in feats]

    df_pos["fantasy_points_scoring"] = df_pos[target_col]
    agg = df_pos.groupby("player_display_name")[feats].sum(numeric_only=True)

    def pct(player: str, feat: str) -> int:
        if feat not in agg.columns or player not in agg.index:
            return 0
        series = agg[feat]
        if series.nunique(dropna=True) <= 1:
            return 50
        rank = series.rank(pct=True).get(player, 0.0)
        return int(round(rank * 100))

    radar = []
    for f in feats:
        label = f.replace("_", " ").title()
        if f == "fantasy_points_scoring":
            label = f"Fantasy Points ({format.upper()})"
        radar.append({
            "subject": label,
            "A": pct(p1_name, f),
            "B": pct(p2_name, f),
            "fullMark": 100
        })

    return radar


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "season": CURRENT_YEAR}


@app.get("/weeks")
def get_weeks():
    return {"weeks": AVAILABLE_WEEKS, "default_week": DEFAULT_WEEK}


@app.get("/players/{position}")
def get_players(position: str):
    df_pos = df_stats[df_stats["position"] == position]
    players = df_pos["player_display_name"].dropna().unique().tolist()
    return {"players": sorted(players)}


@app.get("/compare")
def compare_players(player1: str, player2: str, format: str = "ppr", week: int | None = None):
    """
    Returns:
      - history filtered to data through (week - 1) if week is provided
      - radar computed through (week - 1) if week is provided
    """
    df1 = get_player_data(player1)
    df2 = get_player_data(player2)
    if df1.empty and df2.empty:
        return {}

    target_col = COL_MAP.get(format, "fantasy_points_ppr")

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined["player_name"] = df_combined["player_display_name"]
    df_combined["fantasy_points_ppr"] = df_combined[target_col]

    cols = ["week", "player_name", "fantasy_points_ppr"]
    history_df = df_combined[cols].sort_values(by="week")

    # ✅ Trend history uses cutoff = week - 1 (pre-game view)
    if week is not None:
        wk = clamp_week(int(week))
        cutoff = max(0, wk - 1)
        if cutoff > 0:
            history_df = history_df[history_df["week"] <= cutoff]
        else:
            history_df = history_df.iloc[0:0]

    history = history_df.to_dict(orient="records")

    pos = "QB"
    if not df1.empty:
        pos = str(df1["position"].iloc[0])
    elif not df2.empty:
        pos = str(df2["position"].iloc[0])

    radar = get_radar_data(player1, player2, pos, week, format)

    return {"history": history, "radar": radar}


@app.get("/predict")
def predict_winner(
    player1: str,
    player2: str,
    format: str = "ppr",
    week: int = DEFAULT_WEEK,
):
    """
    Start/Sit for matchup week = week.
    Uses stats only through cutoff = (week - 1).
    """
    target_col = COL_MAP.get(format, "fantasy_points_ppr")
    matchup_week = clamp_week(int(week))
    cutoff = max(0, matchup_week - 1)

    def analyze(name: str) -> Optional[dict]:
        p_stats = get_player_data(name)
        if p_stats.empty:
            return None

        # ✅ Use data only through cutoff (week - 1). If cutoff==0, use empty frame.
        if cutoff > 0:
            p_upto = p_stats[p_stats["week"] <= cutoff].copy()
        else:
            p_upto = p_stats.iloc[0:0].copy()

        games_played = int(len(p_upto))
        recent_n = int(min(3, games_played))

        if games_played == 0:
            avg_points = 0.0
            recent_avg = 0.0

            # still derive stable identity fields from overall data
            p_sorted_all = p_stats.sort_values("week")
            position = str(p_sorted_all["position"].iloc[0])
            team = str(p_sorted_all["recent_team"].iloc[-1])
        else:
            avg_points = safe_float(p_upto[target_col].mean())
            p_sorted = p_upto.sort_values("week")
            recent_avg = safe_float(p_sorted.tail(recent_n)[target_col].mean())

            position = str(p_sorted["position"].iloc[0])
            team = str(p_sorted["recent_team"].iloc[-1])

        # ✅ Opponent/BYE is for the matchup week itself
        opponent, is_home = get_opponent_for_team_week(team, matchup_week)

        reasons: List[str] = []
        trend_bonus = 0.0

        # Trend logic based on cutoff-limited data
        if games_played > 0:
            if recent_avg > avg_points + 1.5:
                trend_bonus = 1.0
                reasons.append(f"Hot form (last {recent_n} game(s) through Week {cutoff}).")
            elif recent_avg < avg_points - 1.5:
                trend_bonus = -0.5
                reasons.append(f"Cold stretch (last {recent_n} game(s) through Week {cutoff}).")

        home_bonus = 1.0 if is_home else 0.0
        if is_home:
            reasons.append("Home-field bump.")

        if opponent == "BYE":
            reasons = ["On BYE this week."]
            projected = 0.0
        else:
            projected = avg_points + trend_bonus + home_bonus

        return {
            "name": name,
            "avg_points": round(avg_points, 2),
            "recent_avg": round(recent_avg, 2),
            "recent_n": recent_n,
            "games_played": games_played,

            "opponent": opponent,
            "is_home": bool(is_home),
            "week": matchup_week,
            "projected_points": round(projected, 2),
            "reasons": reasons,
            "position": position,
            "team": team,
        }

    p1 = analyze(player1)
    p2 = analyze(player2)

    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="Player not found")

    # BYE handling: if one is BYE and the other isn't, the non-BYE wins
    if p1["opponent"] == "BYE" and p2["opponent"] != "BYE":
        winner, loser = p2, p1
    elif p2["opponent"] == "BYE" and p1["opponent"] != "BYE":
        winner, loser = p1, p2
    else:
        winner, loser = (p1, p2) if p1["projected_points"] >= p2["projected_points"] else (p2, p1)

    margin = round(abs(p1["projected_points"] - p2["projected_points"]), 2)
    summary = generate_ai_analysis(winner, loser, format, matchup_week, cutoff)

    return {
        "winner": winner["name"],
        "margin": margin,
        "summary": summary,
        "details": [p1, p2],
        "week": matchup_week,
        "format": format,
        # Optional but useful for debugging/UI text
        "cutoff_week": cutoff,
    }

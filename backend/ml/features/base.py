from __future__ import annotations
import pandas as pd
from typing import Iterable


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if it exists, else a zero series of same length."""
    if name in df.columns:
        return df[name]
    return pd.Series([0.0] * len(df), index=df.index)


def mean_last_n(df: pd.DataFrame, col: str, n: int) -> float:
    s = _col(df, col).tail(n)
    if len(s) == 0:
        return 0.0
    return float(s.mean())


def mean_all(df: pd.DataFrame, col: str) -> float:
    s = _col(df, col)
    if len(s) == 0:
        return 0.0
    return float(s.mean())


def std_all(df: pd.DataFrame, col: str) -> float:
    s = _col(df, col)
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=0))


def count_games(df: pd.DataFrame) -> int:
    return int(len(df))

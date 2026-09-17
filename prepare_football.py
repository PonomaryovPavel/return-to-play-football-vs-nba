"""
Football cohort: Achilles ruptures, ACL tears and meniscus tears among players
of the five big European leagues.

Source: Transfermarkt injury histories as published in salimt/football-datasets
(also on Kaggle as xfkzujqjvx97n/football-datasets, CC0). Transfermarkt editors
record a diagnosis, a start date and the date the player is available again.
Many end dates are rounded: 30 June and 31 December appear far more often than
chance allows, so every result is also checked without them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ACHILLES, ACL, MENISCUS = "Achilles rupture", "ACL tear", "Meniscus tear"
INJURIES = [ACHILLES, ACL, MENISCUS]

TOP5 = {"GB1": "Premier League", "ES1": "LaLiga", "L1": "Bundesliga",
        "IT1": "Serie A", "FR1": "Ligue 1"}

# Same logic as the NBA side: body part and tear in the same record.
# Transfermarkt writes "cruciate ligament" without saying anterior or posterior;
# the posterior ligament is torn far less often, so the label stays "ACL".
TEAR = r"tear|torn|ruptur"
RULES = [(ACHILLES, r"achilles"), (ACL, r"cruciate"), (MENISCUS, r"menisc")]

# Broad version for the sensitivity check: also "injury", "damage", "surgery".
TEAR_BROAD = TEAR + r"|injury|damage|surgery"
EXCLUDE_BROAD = r"strain|partial|stretch|irritation|problems|contusion"

# Records separated by up to this many days describe one absence
# (a tear, then surgery, then "fitness").
GAP_DAYS = 3

# Absences that are not injuries do not extend an injury episode.
NOT_INJURY = re.compile(
    r"corona|covid|quarantine|\bill\b|illness|cold|flu|influenza|virus|infection|"
    r"rest\b|suspen|personal|family|paternity|birth|bereave|red card|yellow card",
    re.I,
)

FIRST_SEASON, LAST_SEASON = 2010, 2019
CUTOFF = pd.Timestamp("2020-03-12")     # leagues stopped within two days of the NBA
HORIZON = 730

# The pages were scraped on 12-13 September 2025: 1,513 of the 1,515 open
# injuries are counted up to those two days. Later end dates are Transfermarkt's
# forecasts, not observed returns.
SNAPSHOT = pd.Timestamp("2025-09-12")


def classify(reason: str, broad: bool = False) -> str | None:
    r = str(reason).lower()
    if broad:
        if re.search(EXCLUDE_BROAD, r) or not re.search(TEAR_BROAD, r):
            return None
    elif not re.search(TEAR, r):
        return None
    for label, pattern in RULES:
        if re.search(pattern, r):
            return label
    return None


def season_year(label: str) -> int | None:
    """'14/15' -> 2014, '99/00' -> 1999."""
    m = re.fullmatch(r"(\d{2})/(\d{2})", str(label).strip())
    if not m:
        return None
    y = int(m.group(1))
    return 2000 + y if y < 60 else 1900 + y


def load_injuries(path: str | Path) -> pd.DataFrame:
    inj = pd.read_csv(path)
    inj = inj.drop_duplicates()
    inj["start"] = pd.to_datetime(inj.from_date, errors="coerce")
    inj["end"] = pd.to_datetime(inj.end_date, errors="coerce")
    inj = inj.dropna(subset=["start", "days_missed"]).copy()
    inj["reason"] = inj.injury_reason.astype(str).str.strip()
    # An open record is counted up to the day the page was scraped.
    inj["scraped"] = inj.start + pd.to_timedelta(inj.days_missed - 1, unit="D")
    # Season from the start date (July to June). Transfermarkt's own label
    # sometimes follows the end of the absence instead.
    inj["season"] = np.where(inj.start.dt.month >= 7, inj.start.dt.year, inj.start.dt.year - 1)
    # Two records with the same start: keep the longer one.
    inj = (inj.sort_values(["player_id", "start", "end"], na_position="last")
              .drop_duplicates(["player_id", "start"], keep="last"))
    return inj.sort_values(["player_id", "start"]).reset_index(drop=True)


def build_episodes(inj: pd.DataFrame) -> pd.DataFrame:
    """Glue back-to-back records of one player into a single absence."""
    inj = inj[~inj.reason.str.contains(NOT_INJURY)].copy()
    rows = []
    for pid, g in inj.groupby("player_id", sort=False):
        current = None
        for r in g.itertuples(index=False):
            r_end = r.end if pd.notna(r.end) else r.scraped
            if current is not None and r.start <= current["reach"] + pd.Timedelta(days=GAP_DAYS):
                current["reasons"].append(r.reason)
                current["open"] = current["open"] or pd.isna(r.end)
                current["reach"] = max(current["reach"], r_end)
                if pd.notna(r.end) and not current["open"]:
                    current["end"] = max(current["end"], r.end)
                current["scraped"] = max(current["scraped"], r.scraped)
                continue
            if current is not None:
                rows.append(current)
            current = {"player_id": pid, "season": r.season, "start": r.start,
                       "end": r.end, "open": pd.isna(r.end), "reach": r_end,
                       "scraped": r.scraped, "reasons": [r.reason],
                       "first_reason": r.reason}
        if current is not None:
            rows.append(current)

    ep = pd.DataFrame(rows)
    ep.loc[ep.open, "end"] = pd.NaT
    ep["reasons"] = ep.reasons.map(lambda rs: " | ".join(rs))
    return ep.drop(columns="reach")


def _label(reasons: str, broad: bool) -> str | None:
    found = {classify(r, broad) for r in reasons.split(" | ")} - {None}
    for label in INJURIES:
        if label in found:
            return label
    return None


def top5_seasons(performances: str | Path) -> pd.DataFrame:
    """Player-seasons with at least one match-day squad in a top-5 league."""
    perf = pd.read_csv(performances, usecols=["player_id", "season_name", "competition_id",
                                              "nb_in_group", "minutes_played"])
    perf = perf[perf.competition_id.isin(TOP5) & (perf.nb_in_group > 0)].copy()
    perf["season"] = perf.season_name.map(season_year)
    perf = perf.sort_values("nb_in_group").drop_duplicates(["player_id", "season"], keep="last")
    perf["league"] = perf.competition_id.map(TOP5)
    return perf[["player_id", "season", "league"]]


def load_births(profiles: str | Path) -> pd.DataFrame:
    pr = pd.read_csv(profiles, usecols=["player_id", "player_name", "date_of_birth", "main_position"])
    pr["dob"] = pd.to_datetime(pr.date_of_birth, errors="coerce")
    return pr[["player_id", "player_name", "dob", "main_position"]]


# A player cleared in June, July or August cannot play a league match before
# the new season. The five leagues start between early and late August, so
# moving such dates to 31 August is the worst case for football: it can only
# make football returns look longer.
SUMMER_MONTHS = (6, 7, 8)
LATEST_RESTART = (8, 31)


def latest_restart(d: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(d) or d.month not in SUMMER_MONTHS:
        return d
    return pd.Timestamp(d.year, *LATEST_RESTART)


def is_rounded(d: pd.Series) -> pd.Series:
    """First or last day of a month: where editors' estimates pile up."""
    return d.notna() & ((d.dt.day == 1) | ((d + pd.Timedelta(days=1)).dt.day == 1))


def cohort(injuries, performances, profiles, first_season=FIRST_SEASON,
           last_season=LAST_SEASON, cutoff=CUTOFF, horizon=HORIZON) -> pd.DataFrame:
    """
    One row per injury in a top-5 league season.

    Follow-up ends at `cutoff`: the March 2020 stoppage for the main
    comparison, the scrape date for recent seasons.
    """
    ep = build_episodes(load_injuries(injuries))
    ep["injury"] = ep.reasons.map(lambda s: _label(s, broad=False))
    ep["injury_broad"] = ep.reasons.map(lambda s: _label(s, broad=True))
    ep = ep[ep.injury_broad.notna()].copy()
    ep = ep[ep.season.between(first_season, last_season)]
    ep = ep.merge(top5_seasons(performances), on=["player_id", "season"], how="inner")
    ep = ep[ep.start < cutoff].copy()
    stop_limit = pd.Series(cutoff, index=ep.index)
    ep["end_late"] = ep.end.map(latest_restart)
    for end_col, suffix in (("end", ""), ("end_late", "_late")):
        end = ep[end_col]
        observed = end.notna() & (end <= stop_limit)
        stop = end.where(observed, stop_limit)
        time = (stop - ep.start).dt.days
        ep["event" + suffix] = (observed & (time < horizon)).astype(int)
        ep["time" + suffix] = time.clip(upper=horizon)
    ep["rounded_end"] = is_rounded(ep.end)

    births = load_births(profiles)
    ep = ep.merge(births, on="player_id", how="left")
    ep["age"] = (ep.start - ep.dob).dt.days / 365.25
    ep["sport"] = "Football"
    ep = ep[ep.time > 0]
    return ep.sort_values(["injury_broad", "start"]).reset_index(drop=True)

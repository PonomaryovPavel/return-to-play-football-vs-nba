"""
NBA cohort: Achilles ruptures, ACL tears and meniscus tears, 2014/15 - 2019/20.

Two sources are combined.

1. The Pro Sports Transactions injury log (scraped by gboogy/nba-injury-data-scraper,
   also on Kaggle as ghopkins/nba-injuries-2010-2018, CC0). It says WHAT happened
   and roughly WHEN: "torn left Achilles tendon (out for season)".
2. Player box scores 2010-2024 (NocturneBear/NBA-Data-2010-2024, MIT). They say
   WHEN the player actually played again.

The two earlier projects built absences from the transaction log alone. Checked
case by case, that method misplaces a large share of the serious injuries: a
waiver is logged like a return, a missing return record glues two injuries
together, and the diagnosis can come from a different note than the tear.
legacy_episodes() keeps the old method so the difference can be shown.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ACHILLES, ACL, MENISCUS = "Achilles rupture", "ACL tear", "Meniscus tear"
INJURIES = [ACHILLES, ACL, MENISCUS]

# Body part and tear must appear in the SAME note. Checked top to bottom,
# first match wins, so an ACL note that also mentions the meniscus is an ACL.
# "repair" counts: a meniscus or tendon is repaired only when it is torn.
TEAR = r"torn|tear|ruptur|repair"
RULES: list[tuple[str, str]] = [
    (ACHILLES, r"achilles"),
    (ACL, r"\bacl\b|anterior cruciate"),
    (MENISCUS, r"menisc"),
]

FIRST_SEASON, LAST_SEASON = 2014, 2019

# The league suspended play on 11 March 2020. Follow-up stops there for
# every sport in this project, so the pandemic calendar never enters the data.
CUTOFF = pd.Timestamp("2020-03-11")
HORIZON = 730

# If the last game before the note is this close, the injury happened in it.
ONSET_WINDOW_DAYS = 3

# A return within this many days of the team's season opener is a return
# "after the summer": the log tells us the player was not ready while his team
# was still playing, and nothing more.
OPENER_WINDOW_DAYS = 21


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

ALIASES = {
    "alex ajinca": "alexis ajinca",
    "enes kanter": "enes freedom",
    "jeff taylor": "jeffery taylor",
    "jose barea": "jj barea",
    "malcom lee": "malcolm lee",
    "nene hilario": "nene",
    "ognen kuzmic": "ognjen kuzmic",
    "ron artest": "metta world peace",
    "wes matthews": "wesley matthews",
}
_DIACRITICS = str.maketrans("çččćšžōéáíúñ", "ccccszoeaiun")


def normalize(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\.?$", "", s).strip()
    s = re.sub(r"[.'`\-]", "", s)
    s = s.translate(_DIACRITICS)
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)


# --------------------------------------------------------------------------
# Raw inputs
# --------------------------------------------------------------------------

def load_events(path: str | Path) -> pd.DataFrame:
    """Transaction log to one row per roster event."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[~(df.Acquired.isna() & df.Relinquished.isna())].copy()
    name = df.Acquired.fillna(df.Relinquished).astype(str)
    df["player"] = (
        name.str.replace(r"\s*\(.*?\)\s*$", "", regex=True).str.strip().str.strip('"')
    )
    df["event"] = np.where(df.Acquired.notna(), "back", "out")
    df["name"] = df.player.map(normalize)
    return df.sort_values(["player", "Date"]).reset_index(drop=True)


def load_box(paths: list[str | Path]) -> pd.DataFrame:
    """Box scores to one row per player per game, with a played flag."""
    cols = ["season_year", "game_date", "gameId", "teamTricode", "teamName", "personId",
            "personName", "comment", "minutes"]
    frames = []
    for p in paths:
        f = pd.read_csv(p, usecols=cols)
        f["playoffs"] = "play_off" in str(p) or "playoffs" in str(p)
        frames.append(f)
    bx = pd.concat(frames, ignore_index=True)
    bx["date"] = pd.to_datetime(bx.game_date)
    bx["season"] = bx.season_year.str.slice(0, 4).astype(int)
    comment = bx.comment.fillna("").str.strip()
    mins = bx.minutes.fillna("").astype(str).str.strip()
    bx["played"] = (comment == "") & ~mins.isin(["", "0:00", "00:00", "0"])
    bx["name"] = bx.personName.map(normalize)
    return bx.drop(columns=["game_date", "comment", "minutes"]).sort_values("date")


def load_ages(path: str | Path) -> pd.DataFrame:
    st = pd.read_csv(path, usecols=["PLAYER_NAME", "AGE", "SEASON"])
    st["season"] = st.SEASON.astype(str).str.slice(0, 4).astype(int)
    st["name"] = st.PLAYER_NAME.map(normalize)
    return st.groupby(["name", "season"], as_index=False).AGE.max()


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------

def classify_note(note: str) -> str | None:
    note = str(note).lower()
    if not re.search(TEAR, note):
        return None
    for label, pattern in RULES:
        if re.search(pattern, note):
            return label
    return None


def injury_notes(events: pd.DataFrame) -> pd.DataFrame:
    out = events[events.event == "out"].copy()
    out["injury"] = out.Notes.map(classify_note)
    return out[out.injury.notna()][["player", "name", "Date", "Team", "injury", "Notes"]]


def _season_calendar(box: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """First and last game of every team in every season, and the league's
    last regular-season day."""
    teams = box.groupby(["season", "teamTricode"]).date.agg(first="min", last="max").reset_index()
    rs_end = box[~box.playoffs].groupby("season").date.max()
    return teams, rs_end


def build_cohort(events: pd.DataFrame, box: pd.DataFrame,
                 first_season: int = FIRST_SEASON, last_season: int = LAST_SEASON) -> pd.DataFrame:
    """
    One row per injury.

    onset     - the game in which the player got hurt, or the note date if
                there was no game in the three days before it
    ret       - first game he played after onset
    lower     - for returns right after a summer: the last game his team
                played before that summer (he was not ready by then, and the
                data cannot say more). Otherwise equal to ret.
    """
    teams, rs_end = _season_calendar(box)
    first_game = teams.set_index(["season", "teamTricode"])["first"]
    last_game = teams.set_index(["season", "teamTricode"])["last"]

    notes = injury_notes(events)
    ids = box.groupby("name").personId.nunique()
    rows = []
    for name, g in notes.groupby("name"):
        if name not in ids.index:
            continue
        games = box[box.name == name]
        if ids[name] > 1:
            # Two players share a name: keep the one who played for the team in the note.
            teams_in_notes = set(g.Team)
            match = games[games.teamName.isin(teams_in_notes)].personId.unique()
            if len(match) != 1:
                continue
            games = games[games.personId == match[0]]
        played = games[games.played]

        current = None
        seen: list[tuple[pd.Timestamp, str]] = []
        for note in g.sort_values("Date").itertuples(index=False):
            # "Recovering from surgery" after a comeback is rest management of an
            # injury already counted, not a new tear.
            if "recovering" in str(note.Notes).lower() and any(
                    kind == note.injury and (note.Date - when).days <= 365 for when, kind in seen):
                continue
            seen.append((note.Date, note.injury))
            if current is not None:
                back_between = played[(played.date > current["onset"]) & (played.date < note.Date)]
                if back_between.empty:
                    # Same absence. A more serious finding upgrades the label.
                    if INJURIES.index(note.injury) < INJURIES.index(current["injury"]):
                        current["injury"] = note.injury
                    continue
                rows.append(current)

            before = played[played.date <= note.Date]
            if before.empty:
                current = None          # hurt before his first NBA game
                continue
            last = before.iloc[-1]
            close = (note.Date - last.date).days <= ONSET_WINDOW_DAYS
            onset = last.date if close else note.Date
            current = {
                "player": note.player, "name": name, "injury": note.injury,
                "note_date": note.Date, "onset": onset, "note": note.Notes,
                "team_before": last.teamTricode, "season_before": last.season,
            }
        if current is not None:
            rows.append(current)

    ep = pd.DataFrame(rows)
    # A new league year starts in July: an August injury belongs to the coming season.
    ep["season"] = ep.onset.map(lambda d: d.year if d.month >= 7 else d.year - 1)
    ep = ep[ep.season.between(first_season, last_season) & (ep.onset < CUTOFF)].copy()

    rets, rteams, rseasons = [], [], []
    for r in ep.itertuples(index=False):
        p = box[(box.name == r.name) & (box.date > r.onset) & box.played]
        rets.append(p.date.iloc[0] if len(p) else pd.NaT)
        rteams.append(p.teamTricode.iloc[0] if len(p) else None)
        rseasons.append(p.season.iloc[0] if len(p) else None)
    ep["ret"] = rets
    ep["team_after"], ep["season_after"] = rteams, rseasons

    lowers = []
    for r in ep.itertuples(index=False):
        lower = r.ret
        # Only returns seen within follow-up get a bound; a later return
        # stays censored at the cutoff in both versions.
        if pd.notna(r.ret) and r.ret <= CUTOFF and r.season_after > r.season_before:
            opener = first_game.get((r.season_after, r.team_after))
            if opener is not None and (r.ret - opener).days <= OPENER_WINDOW_DAYS:
                prev = r.season_after - 1
                team = _team_in_season(box, r.name, prev)
                if team is None and prev == r.season_before:
                    team = r.team_before
                # His team's last game if we know the team, else the league's
                # last regular-season day, which no team finishes before.
                bound = last_game.get((prev, team)) if team else rs_end.get(prev)
                if bound is not None and bound > r.onset:
                    lower = bound
        lowers.append(lower)
    ep["lower"] = lowers
    return ep.reset_index(drop=True)


def _team_in_season(box: pd.DataFrame, name: str, season: int) -> str | None:
    """The last team that listed the player in a season, injured or not."""
    rows = box[(box.name == name) & (box.season == season)]
    return rows.teamTricode.iloc[-1] if len(rows) else None


def to_survival(ep: pd.DataFrame, end_col: str, cutoff=CUTOFF, horizon=HORIZON) -> pd.DataFrame:
    """time/event columns for a chosen end date, censored at cutoff and horizon."""
    end = ep[end_col]
    observed = end.notna() & (end <= cutoff)
    stop = end.where(observed, cutoff)
    time = (stop - ep.onset).dt.days
    return pd.DataFrame({
        "time": time.clip(upper=horizon),
        "event": (observed & (time < horizon)).astype(int),
    }, index=ep.index)


def attach_age(ep: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    """Age in the injury season; if that season is missing, last season's age + 1."""
    same = ages.rename(columns={"AGE": "age_same"})
    prev = ages.assign(season=ages.season + 1, AGE=ages.AGE + 1).rename(columns={"AGE": "age_prev"})
    out = ep.merge(same, on=["name", "season"], how="left")
    out = out.merge(prev, on=["name", "season"], how="left")
    out["age"] = out.age_same.fillna(out.age_prev)
    return out.drop(columns=["age_same", "age_prev"])


def cohort(injury_log, box_paths, stats,
           first_season: int = FIRST_SEASON, last_season: int = LAST_SEASON) -> pd.DataFrame:
    """The analysis table: one row per injury, with three versions of the outcome."""
    events = load_events(injury_log)
    box = load_box(box_paths)
    ep = attach_age(build_cohort(events, box, first_season, last_season), load_ages(stats))

    s = to_survival(ep, "ret")
    ep["time"], ep["event"] = s.time, s.event
    s = to_survival(ep, "lower")
    ep["time_opt"], ep["event_opt"] = s.time, s.event
    ep["summer_return"] = (ep.event == 1) & (ep.time_opt < ep.time)

    ep = ep[ep.time > 0].copy()
    ep["sport"], ep["league"] = "NBA", "NBA"
    ep["start"] = ep.onset
    return ep.sort_values(["injury", "onset"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# The method used in nba-injury-cost and nba-return-after-injury
# --------------------------------------------------------------------------

LEGACY_SEVERE = r"torn|tear|ruptur|surgery|repair"
LEGACY_RULES = [(ACHILLES, r"achilles", True), (ACL, r"\bacl\b|anterior cruciate", False),
                (MENISCUS, r"meniscus", False)]
LEGACY_NOT_INJURY = (r"illness|\brest\b|personal|flu\b|virus|suspend|"
                     r"conditioning|g league|assign|maternity|bereav")
LEGACY_DATA_END = pd.Timestamp("2020-10-06")


def _legacy_in_season_days(start, end):
    if pd.isna(start) or pd.isna(end) or end < start:
        return np.nan
    total = 0
    for year in range(start.year - 1, end.year + 1):
        lo, hi = pd.Timestamp(year, 10, 15), pd.Timestamp(year + 1, 6, 15)
        a, b = max(start, lo), min(end, hi)
        if b > a:
            total += (b - a).days
    return total


def legacy_cohort(events: pd.DataFrame) -> pd.DataFrame:
    """
    The Achilles, ACL and meniscus episodes exactly as the previous project
    analysed them: absences built from the log alone (the first "out" opens an
    episode, the next "back" closes it) and the diagnosis searched in all notes
    of the absence together. Seasons 2014/15 - 2019/20.
    """
    episodes = []
    for player, group in events.groupby("player", sort=False):
        current = None
        for row in group.itertuples(index=False):
            if row.event == "out":
                if current is None:
                    current = {"player": player, "start": row.Date, "notes": [str(row.Notes)]}
                else:
                    current["notes"].append(str(row.Notes))
            elif current is not None:
                episodes.append({**current, "end": row.Date})
                current = None
        if current is not None:
            episodes.append({**current, "end": pd.NaT})
    ep = pd.DataFrame(episodes)
    ep["text"] = ep.notes.map(lambda n: " | ".join(n).lower())

    def classify(text):
        for label, pattern, needs in LEGACY_RULES:
            if re.search(pattern, text):
                if needs and not re.search(LEGACY_SEVERE, text):
                    return None
                return label
        return None

    ep["injury"] = ep.text.map(classify)
    ep = ep[ep.injury.notna() & ~ep.text.str.contains(LEGACY_NOT_INJURY, regex=True)].copy()
    ep["season"] = ep.start.map(lambda d: d.year if d.month >= 10 else d.year - 1)
    ep["event"] = ep.end.notna().astype(int)
    ep["days"] = [_legacy_in_season_days(a, b)
                  for a, b in zip(ep.start, ep.end.fillna(LEGACY_DATA_END))]
    ep = ep[(ep.days > 0) & (ep.season >= FIRST_SEASON)].copy()
    ep["name"] = ep.player.map(normalize)
    return ep.drop(columns="notes").sort_values(["injury", "start"]).reset_index(drop=True)


def audit_legacy(legacy: pd.DataFrame, rebuilt: pd.DataFrame, box: pd.DataFrame,
                 tolerance: int = 14) -> pd.DataFrame:
    """
    Check each old episode against the game logs.

    correct              - same tear, start and return both within two weeks
    not a tear           - the label came from a sprain, tendinitis or another body part
    before NBA debut     - the tear happened before the player's first NBA game
    wrong start          - the absence opens at an earlier, unrelated note
    counted twice        - a second episode for a tear already counted
    return that was not  - the logged "return" was a waiver or a trade
    wrong return date    - the player came back, but not on the logged day
    """
    debut = box[box.played].groupby("name").date.min()
    rows, seen = [], set()
    for r in legacy.itertuples(index=False):
        same = rebuilt[(rebuilt.name == r.name) & (rebuilt.injury == r.injury)]
        near = same[(same.onset - r.start).abs().dt.days <= tolerance]
        match = None
        if same.empty:
            first = debut.get(r.name)
            verdict = ("before NBA debut" if first is None or first > r.start - pd.Timedelta(days=tolerance)
                       else "not a tear")
        elif near.empty:
            match = same.iloc[(same.onset - r.start).abs().argmin()]
            verdict = "counted twice" if (r.name, match.onset) in seen else "wrong start"
        else:
            match = near.iloc[0]
            if (r.name, match.onset) in seen:
                verdict = "counted twice"
            else:
                ret = match.ret if pd.notna(match.ret) and match.ret <= LEGACY_DATA_END else pd.NaT
                end = r.end if pd.notna(r.end) and r.end <= LEGACY_DATA_END else pd.NaT
                if pd.isna(end) and pd.isna(ret):
                    verdict = "correct"
                elif pd.notna(end) and pd.notna(ret) and abs((end - ret).days) <= tolerance:
                    verdict = "correct"
                elif pd.notna(end) and (pd.isna(ret) or ret > end + pd.Timedelta(days=tolerance)):
                    verdict = "return that was not"
                else:
                    verdict = "wrong return date"
        if match is not None:
            seen.add((r.name, match.onset))
        rows.append({
            "player": r.player, "injury": r.injury,
            "old_start": r.start.date(),
            "old_return": None if pd.isna(r.end) else r.end.date(),
            "verdict": verdict,
            "onset": None if match is None else match.onset.date(),
            "first_game_back": None if match is None or pd.isna(match.ret) else match.ret.date(),
            "log_text": r.text,
        })
    return pd.DataFrame(rows)

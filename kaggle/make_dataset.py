"""
Build the Kaggle dataset: every NBA Achilles, ACL and meniscus tear of
2010/11 - 2019/20 with the first game played afterwards, plus the audit of
the transaction-log method. Run from the repository root after get_data.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prepare_nba as NB  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "kaggle" / "dataset"

DESCRIPTION = """\
Every Achilles rupture, ACL tear and meniscus tear logged for NBA players from 2010/11 to 2019/20, with the first game each player played afterwards.

The injury comes from the Pro Sports Transactions log (also on Kaggle as ghopkins/nba-injuries-2010-2018). A tear counts only when one note names both the body part and the tear. The return comes from player box scores 2010-2024 (NocturneBear/NBA-Data-2010-2024, MIT), not from the log: in the log, waivers look like comebacks and missing records glue separate absences together.

### nba_serious_tears_2010_2020.csv

| column | meaning |
|---|---|
| player, injury, season, age | who, what, NBA season (July to June), age that season |
| log_date, log_note | first log entry that names the tear |
| onset | the game in which the player got hurt, or log_date if he had not played in the three days before |
| team_at_injury, team_at_return | team abbreviations from the box scores |
| first_game_back | first regular-season or playoff game played after onset, up to June 2024 |
| days_to_first_game | days from onset to first_game_back |
| returned_by_2020_03_11 | whether the return came before the NBA suspended play |
| summer_return | returned within 21 days of his team's opener, before March 2020 |
| earliest_possible_return | for summer returns, his team's last game of the previous season; otherwise first_game_back |

Tears before a player's first NBA game are left out. A return after October 2020 can include later injuries that the log no longer covers: Klay Thompson tore his Achilles in November 2020 while recovering from his ACL.

### transaction_log_method_audit.csv

The 54 Achilles, ACL and meniscus episodes that a transaction-log-only method produced for 2014/15 to 2019/20, each with a verdict against the box scores. 23 hold up.

Analysis and code: https://github.com/PonomaryovPavel/return-to-play-football-vs-nba
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    boxes = [RAW / f"nba_box_regular_{i}.csv" for i in (1, 2, 3)] + [RAW / "nba_box_playoffs.csv"]
    nba = NB.cohort(ROOT / "data" / "nba_injuries_2010-2020.csv", boxes,
                    RAW / "nba_player_stats_1996_2023.csv", first_season=2010)

    table = nba.rename(columns={
        "note_date": "log_date", "note": "log_note", "ret": "first_game_back",
        "lower": "earliest_possible_return", "team_before": "team_at_injury",
        "team_after": "team_at_return",
    })
    table["days_to_first_game"] = (table.first_game_back - table.onset).dt.days
    table["returned_by_2020_03_11"] = table.first_game_back.le(NB.CUTOFF)
    cols = ["player", "injury", "season", "age", "log_date", "log_note", "onset",
            "team_at_injury", "first_game_back", "team_at_return", "days_to_first_game",
            "returned_by_2020_03_11", "summer_return", "earliest_possible_return"]
    table = table[cols].sort_values(["injury", "onset"])
    for c in ("log_date", "onset", "first_game_back", "earliest_possible_return"):
        table[c] = table[c].dt.strftime("%Y-%m-%d")
    table.to_csv(OUT / "nba_serious_tears_2010_2020.csv", index=False)

    events = NB.load_events(ROOT / "data" / "nba_injuries_2010-2020.csv")
    box = NB.load_box(boxes)
    audit = NB.audit_legacy(NB.legacy_cohort(events), nba, box)
    audit.to_csv(OUT / "transaction_log_method_audit.csv", index=False)

    meta = {
        "title": "NBA ACL, Achilles and Meniscus Tears 2010-2020",
        "subtitle": "100 tears with the first game played afterwards, checked against box scores",
        "id": "ponomaryovpavel/nba-acl-achilles-meniscus-tears-2010-2020",
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["basketball", "health", "sports"],
        "description": DESCRIPTION,
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(table)} tears, {len(audit)} audited episodes -> {OUT}")


if __name__ == "__main__":
    main()

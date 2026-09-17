"""
Download the raw tables the analysis needs, pinned to exact commits.

Football: Transfermarkt tables published in salimt/football-datasets.
NBA games: player box scores 2010-2024 from NocturneBear/NBA-Data-2010-2024 (MIT).
NBA ages: season-level player stats shipped with nba-return-after-injury.
The NBA injury log itself is small and lives in data/ already.

Each file is checked against a SHA-256 hash, so a silent upstream change
cannot slip into the results.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"

FOOTBALL = "4701b2bb96b4c26817c35d41d4e1ba940fb04832"
NBA = "5949333165ad0ef6f978b61e0bdc00a2d5cc8ab3"
BOX = "a5f108b5b1f08074d78b9e8e901926a9ce4c06c5"
TM = "datalake/transfermarkt"
BOX_URL = f"https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/{BOX}"

FILES = {
    "player_injuries.csv": (
        f"https://raw.githubusercontent.com/salimt/football-datasets/{FOOTBALL}/"
        f"{TM}/player_injuries/player_injuries.csv",
        "fab7edd70cd4be43c682172a3d7ac8c63dc8e119cd2c90ec2acd8a8f801bf192",
    ),
    "player_profiles.csv": (
        f"https://raw.githubusercontent.com/salimt/football-datasets/{FOOTBALL}/"
        f"{TM}/player_profiles/player_profiles.csv",
        "7ef9afbacf97577d099f21a0ef0ea0d46381bd4806a119f095b9562762a20096",
    ),
    # Stored with Git LFS, so it comes from the media host, not the raw one.
    "player_performances.csv": (
        f"https://media.githubusercontent.com/media/salimt/football-datasets/{FOOTBALL}/"
        f"{TM}/player_performances/player_performances.csv",
        "21eda6556654478fe986de8972d3b94aea6551ebcd74d5eb153b65edb53d132b",
    ),
    "nba_box_regular_1.csv": (
        f"{BOX_URL}/regular_season_box_scores_2010_2024_part_1.csv",
        "ca9636d08c577e3d752f69e32f90b173f4c3374c15f4dc8494b655049fdfc557",
    ),
    "nba_box_regular_2.csv": (
        f"{BOX_URL}/regular_season_box_scores_2010_2024_part_2.csv",
        "5643af8a9f15ce3540dced759375ca79e77c71644e23270136f23b7a57fc182d",
    ),
    "nba_box_regular_3.csv": (
        f"{BOX_URL}/regular_season_box_scores_2010_2024_part_3.csv",
        "65a01fee3df865a272e578cf4b365dce7dbc62eef2ce6f15f0844cf6694ef375",
    ),
    "nba_box_playoffs.csv": (
        f"{BOX_URL}/play_off_box_scores_2010_2024.csv",
        "d60906641c67026c167c1f95b8ccc1e61a8c3e4d0525eaa1f4be4329cd1ea498",
    ),
    "nba_player_stats_1996_2023.csv": (
        f"https://raw.githubusercontent.com/PonomaryovPavel/nba-return-after-injury/{NBA}/"
        "player_stats_rs_1996_2023.csv",
        "ae463112616fec8d1490b1cda91c7b16523498aeb2683191380a04c7ab73b9dc",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, url: str, expected: str) -> Path:
    target = RAW / name
    if target.exists() and sha256(target) == expected:
        print(f"ok        {name}")
        return target

    print(f"download  {name}")
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)

    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        raise RuntimeError(f"{name}: sha256 {actual} does not match the pinned {expected}")
    tmp.replace(target)
    return target


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        fetch(name, url, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())

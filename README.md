# Back on the pitch, back on the court

How long elite athletes are out after an ACL tear, an Achilles rupture or a meniscus tear: the five big European football leagues against the NBA, seasons 2010/11 to 2019/20. Open data, reproducible Python.

Half of the footballers are back 212 days after an ACL tear; for NBA players it takes 359 days. After an Achilles rupture the medians are 183 and 357 days. The gap narrows but holds when the summer break is pushed to its extremes for both sports. For meniscus tears the data cannot tell the sports apart.

[Русская версия](README.ru.md)

![Share of players back after injury, football against the NBA](figures/01_return_curves.png)

| Injury (football / NBA tears) | Football, recorded | NBA, recorded | Football, bound | NBA, bound |
|---|---:|---:|---:|---:|
| ACL tear (306 / 30) | 212 (196–217) | 359 (324–446) | 219 (212–228) | 331 (264–377) |
| Achilles rupture (45 / 22) | 183 (158–205) | 357 (299–588) | 190 (174–220) | 339 (218–399) |
| Meniscus tear (50 / 48) | 71 (56–99) | 115 (56–193) | 83 (56–107) | 53 (47–69) |

Median days from injury to return, 95% bootstrap interval in brackets. Within a year of an ACL tear 92.6% of the footballers are back and 53.7% of the NBA players (68.1% at the bound). The Achilles figures are 97.5% and 64.3% (68.6%).

## How the comparison works

- Football: Transfermarkt injury histories of players in a match-day squad of the Premier League, LaLiga, Bundesliga, Serie A or Ligue 1 in the season of the injury. The injury ends on the Transfermarkt end date.
- NBA: the Pro Sports Transactions injury log says what was torn and when; player box scores say when the player first played again.
- A record counts only if the body part and a tear are named in the same note.
- Follow-up stops in March 2020, when both sports suspended play. Players not back by then are censored, so the curves are Kaplan–Meier estimates.
- The calendar bound is an extreme case for each sport. An NBA player who returns at a season opener might have been ready long before; his return is moved back to his team's last game of the previous season. A footballer cleared in June, July or August is moved to 31 August. Both moves can only shrink the gap.

![Medians with intervals](figures/02_medians.png)

## The NBA calendar

None of the 35 NBA comebacks from an ACL or Achilles tear came between March and September, and 28 came in October to December. Football returns are spread across the year. Without the bound, part of the NBA figure would be the summer break, not the injury.

![Month of return](figures/03_return_months.png)

## A correction to the previous project

[nba-return-after-injury](https://github.com/PonomaryovPavel/nba-return-after-injury) built absences from the transaction log alone. Checked against box scores, 23 of its 54 Achilles, ACL and meniscus episodes hold up. The rest break down as follows:

| What went wrong | Episodes |
|---|---:|
| return logged on the wrong day (26 to 190 days off) | 12 |
| a waiver or similar roster move logged as a comeback | 6 |
| absence opened by an earlier, unrelated note | 5 |
| not a tear at all | 4 |
| tear before the player's NBA debut | 3 |
| the same tear counted twice | 1 |

Every episode with its verdict is in [results/nba_old_episodes_audit.csv](results/nba_old_episodes_audit.csv). The figures for these three injuries in that project should not be used, and [nba-injury-cost](https://github.com/PonomaryovPavel/nba-injury-cost) relies on the same method.

![Audit of the old method](figures/04_old_method_audit.png)

## Checks

- Hazard ratios, adjusted for age, at the bounds: 2.54 (1.63–3.97) for the ACL and 3.11 (1.56–6.20) for the Achilles; above 1 means footballers come back sooner. Age shows no effect. The ratios work as a test only: the proportional-hazards assumption does not hold between the sports.
- Transfermarkt rounds end dates: 30 June appears 8 times and 31 December 6 times as often as an average day. Without rounded dates the football ACL and Achilles medians move by one day, with broader labels by two and three days, and without goalkeepers not at all.
- Premier League players stay out longer after an ACL tear than those in the other four leagues: 243 against 197 days (p = 0.00004). The data cannot say whether that is medicine or editing habits.
- In 2021/22 to 2024/25 the football ACL median is 245 days, a month longer than in the 2010s (p = 0.00013). The Achilles change, 183 to 204 days from 21 ruptures, is within chance.
- Published figures agree. A 2025 meta-analysis gives 367 days for NBA players after ACL reconstruction ([D'Ambrosi et al.](https://www.sciencedirect.com/science/article/pii/S2059775425006418)); the UEFA Elite Club Injury Study gives 6.6 months to training and 7.4 months to matches after an ACL tear in football ([Waldén et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27034129/)).

![Football then and now](figures/05_football_then_and_now.png)

## What this does not show

- Why the gap exists. Landing loads on a hard court, contract protection and league calendars are candidates; none is tested here.
- The end points differ: Transfermarkt records availability, the NBA side the first game played. In UEFA data that step takes under a month.
- A return is not a recovery. Nothing here measures whether a player reached his previous level.
- The NBA sample is small: its direction holds under every check, its exact day counts do not.
- Transfermarkt does not say which cruciate ligament was torn. Posterior cruciate injuries are rare in elite football ([28 in 17 seasons, median lay-off 31 days](https://doaj.org/article/72b96066724a499a8e8f3049a89af6b0)), so they can only flatter football slightly.
- A torn meniscus can be trimmed or repaired, and the two take months apart; the sources rarely say which. The calendar bound also flips the order of the sports, so there is no verdict for the meniscus.

## Files

| File | What it is |
|---|---|
| `analysis.ipynb` | the full analysis with outputs; start here |
| `get_data.py` | downloads the raw tables at pinned commits and checks SHA-256 |
| `prepare_nba.py` | NBA cohort from the injury log and box scores, plus the audit of the old method |
| `prepare_football.py` | football cohort from Transfermarkt histories |
| `survival.py` | Kaplan–Meier, bootstrap, log-rank, Cox |
| `figures.py` | charts |
| `results/` | summary tables written by the notebook |
| `data/nba_injuries_2010-2020.csv` | the NBA injury log |

## Reproduce

```bash
git clone https://github.com/PonomaryovPavel/return-to-play-football-vs-nba.git
cd return-to-play-football-vs-nba
pip install -r requirements.txt
python get_data.py
jupyter lab analysis.ipynb
```

## Data

- Football: Transfermarkt injury histories, profiles and appearances as collected by [salimt/football-datasets](https://github.com/salimt/football-datasets), also published on Kaggle under CC0. The original data belongs to Transfermarkt, so it is downloaded at run time and not stored here. Pages were scraped on 12–13 September 2025.
- NBA injuries: [Pro Sports Transactions](https://www.prosportstransactions.com/) via [gboogy/nba-injury-data-scraper](https://github.com/gboogy/nba-injury-data-scraper), on Kaggle as [NBA Injuries from 2010-2020](https://www.kaggle.com/datasets/ghopkins/nba-injuries-2010-2018) (CC0).
- NBA box scores: [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024) (MIT).
- NBA ages: season stats from [Brescou/NBA-dataset-stats-player-team](https://github.com/Brescou/NBA-dataset-stats-player-team), as bundled with the previous project.

Code is under the MIT licence.

Earlier work in the series: [nba-injury-cost](https://github.com/PonomaryovPavel/nba-injury-cost), [nba-return-after-injury](https://github.com/PonomaryovPavel/nba-return-after-injury).

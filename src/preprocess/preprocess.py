import numpy as np
import pandas as pd
from sklearn import preprocessing

"""This file contains relevant functions for getting and preprocessing barttorvik
data.

For example, to get 2024 season data:
df, X, y = get_data(2024)
"""

RES_MAP = {"W": 1, "L": 0}  # map categorical to numeric
VEN_MAP = {"H": 1, "N": 0, "A": -1}
STATS_COLUMNS = [
    "AdjO",
    "AdjD",
    "OffEfg%",
    "OffTo%",
    "OffReb%",
    "OffFTR",
    "DefEfg%",
    "DefTo%",
    "DefReb%",
    "DefFTR",
]  # which features to include


# given a year, gets relevant barttorvik game data with rolling averages, producing a df along with X and y np arrays
def get_data(year):
    try:
        url = "https://barttorvik.com/getgamestats.php?year=" + str(year) + "&csv=1"
    except:
        raise ValueError("Invalid year: " + str(year))
    df = pd.read_csv(url, header=None)
    # add column headers
    column_headers = {
        0: "Date",
        1: "Type",
        2: "Team",
        3: "Team Conf",
        4: "Opp",
        5: "Venue",
        6: "Result",
        7: "AdjO",
        8: "AdjD",
        9: "OffEff",
        10: "OffEfg%",
        11: "OffTo%",
        12: "OffReb%",
        13: "OffFTR",
        14: "DefEff",
        15: "DefEfg%",
        16: "DefTo%",
        17: "DefReb%",
        18: "DefFTR",
        19: "G-SC",
        20: "Opp Conf",
        21: "Num",
        22: "Year",
        23: "T",
        24: "Title",
        25: "Team Coach",
        26: "Opp Coach",
        27: "Diff",
        28: "Diff2",
        29: "List",
        30: "End",
    }
    df.rename(columns=column_headers, inplace=True)
    # sort by date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%m/%d/%y")
    df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)
    # separate result into W/L, home score, away score columns
    result_split = df["Result"].str.extract(r"([WL]), (\d+)-(\d+)")
    df.insert(6, "Away Score", result_split[2].astype(int))
    df.insert(6, "Home Score", result_split[1].astype(int))
    df.insert(6, "Win", result_split[0])
    # convert result to binary: 1 for W, 0 for L
    df["Win"] = df["Win"].apply(lambda x: RES_MAP[x])
    # convert venue to numerical: 1 for H, 0 for N, -1 for A
    df["Venue"] = df["Venue"].apply(lambda x: VEN_MAP[x])
    # get rolling averages
    num_games = 10
    for stat in STATS_COLUMNS:
        df[f"Team_{stat}_avg"] = (
            df.groupby("Team")[stat]
            .rolling(num_games, min_periods=1)
            .mean()
            .shift(1)  # Exclude the current game
            .reset_index(level=0, drop=True)
        )
    # Compute rolling averages for the opponent
    for stat in STATS_COLUMNS:
        df[f"Opponent_{stat}_avg"] = (
            df.groupby("Opp")[stat]
            .rolling(5, min_periods=1)
            .mean()
            .shift(1)  # Exclude the current game
            .reset_index(level=0, drop=True)
        )
    df.dropna(inplace=True)
    # get stat differentials
    for stat in STATS_COLUMNS:
        df[f"{stat}_diff"] = df[f"Team_{stat}_avg"] - df[f"Opponent_{stat}_avg"]
    features = [
        "Venue",
        "AdjO_diff",
        "AdjD_diff",
        "OffEfg%_diff",
        "OffTo%_diff",
        "OffReb%_diff",
        "OffFTR_diff",
        "DefEfg%_diff",
        "DefTo%_diff",
        "DefReb%_diff",
        "DefFTR_diff",
    ]
    X = df[features].to_numpy()
    X = preprocessing.scale(X)
    y = df["Win"].to_numpy()
    y = np.reshape(y, (-1, 1))

    return df, X, y


# get_data(2025)

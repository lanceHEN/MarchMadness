from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class AbstractModel(ABC):

    def __init__(self, url):
        df = pd.read_csv(url, header=None)
        # add column headers
        column_headers = {
            0: "Date", 1: "Type", 2: "Team", 3: "Team Conf", 4: "Opp",
            5: "Venue", 6: "Result", 7: "AdjO", 8: "AdjD", 9: "OffEff",
            10: "OffEfg%", 11: "OffTo%", 12: "OffReb%", 13: "OffFTR", 14: "DefEff",
            15: "DefEfg%", 16: "DefTo%", 17: "DefReb%", 18: "DefFTR", 19: "G-SC",
            20: "Opp Conf", 21: "Num", 22: "Year", 23: "T", 24: "Title",
            25: "Team Coach", 26: "Opp Coach", 27: "Diff", 28: "Diff2", 29: "List", 30: "End"
        }
        df.rename(columns=column_headers, inplace=True)
        # sort by date
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', format='%m/%d/%y')
        df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        # separate result into W/L, home score, away score columns
        result_split = df['Result'].str.extract(r'([WL]), (\d+)-(\d+)')
        df.insert(6, 'Away Score', result_split[2].astype(int))
        df.insert(6, 'Home Score', result_split[1].astype(int))
        df.insert(6, 'Win', result_split[0])
        # convert result to binary: 1 for W, 0 for L
        self.res_map = {'W':1, 'L':0}
        df['Win'] = df['Win'].apply(lambda x: self.res_map[x])
        # convert venue to numerical: 1 for H, 0 for N, -1 for A
        self.ven_map = {'H':1, 'N':0, 'A':-1}
        df['Venue'] = df['Venue'].apply(lambda x: self.ven_map[x])
        # get rolling averages
        self.stats_columns = ['AdjO', 'AdjD', 'OffEfg%', 'OffTo%', 'OffReb%', 'OffFTR', 'DefEfg%', 'DefTo%', 'DefReb%', 'DefFTR']
        num_games = 10
        for stat in self.stats_columns:
            df[f'Team_{stat}_avg'] = (
                df.groupby('Team')[stat]
                .rolling(num_games, min_periods=1)
                .mean()
                .shift(1)  # Exclude the current game
                .reset_index(level=0, drop=True)
        )
        # Compute rolling averages for the opponent
        for stat in self.stats_columns:
            df[f'Opponent_{stat}_avg'] = (
                df.groupby('Opp')[stat]
                .rolling(5, min_periods=1)
                .mean()
                .shift(1)  # Exclude the current game
                .reset_index(level=0, drop=True)
        )
        df.dropna(inplace=True)
        # get stat differentials
        for stat in self.stats_columns:
            df[f'{stat}_diff'] = df[f'Team_{stat}_avg'] - df[f'Opponent_{stat}_avg']
        self.df = df
        features = ['Venue', 'AdjO_diff', 'AdjD_diff', 'OffEfg%_diff', 'OffTo%_diff', 'OffReb%_diff', 'OffFTR_diff',
                    'DefEfg%_diff', 'DefTo%_diff', 'DefReb%_diff', 'DefFTR_diff']
        X = df[features].to_numpy()
        y = df['Win'].to_numpy()
        self.y = np.reshape(y,(-1,1))
        X1 = X[y.flatten()==1]
        X0 = X[y.flatten()==0]
        u = np.reshape(np.mean(X0, axis=0) - np.mean(X1, axis=0), (X.shape[1],1))
        S1 = u.dot(u.T)
        C0 = np.eye(X0.shape[0]) - 1/X0.shape[0] * np.ones((X0.shape[0],X0.shape[0]))
        C1 = np.eye(X1.shape[0]) - 1/X1.shape[0] * np.ones((X1.shape[0],X1.shape[0]))
        S2 = X0.T.dot(C0.dot(X0)) + X1.T.dot(C1.dot(X1))
        Q = np.linalg.inv(S2).dot(S1)
        most_important_idx = np.argmax(np.linalg.eig(Q)[0])
        self.vhat = np.reshape(np.linalg.eig(Q)[1][:,most_important_idx],(-1,1))
        #project data
        alphas = X.dot(self.vhat)
        self.alphas = np.real(alphas)

    @abstractmethod
    def train(self):
        pass

    def _compute_rolling_average(self, team, window=10):
        team_games = self.df[self.df['Team'] == team]
        rolling_avg = {}

        for stat in self.stats_columns:
            rolling_avg[f'Team_{stat}_avg'] = team_games[stat].tail(window).mean()

        return rolling_avg

    @abstractmethod
    def predict(self, team1, team2, venue):
        pass
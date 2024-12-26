import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model.AbstractModel import AbstractModel

class MMLogisticRegression(AbstractModel):

    def __init__(self, url):
        super().__init__(url)
        self.log = LogisticRegression()

    def train(self):
        self.log.fit(self.alphas, np.ravel(self.y))

    def predict(self, team1, team2, venue):
        # obtain rolling averages
        team_rolling_avg = self._compute_rolling_average(team1.get_name())
        opp_rolling_avg = self._compute_rolling_average(team2.get_name())
        # obtain x vector
        x = [self.ven_map[venue]]
        for stat in self.stats_columns:
            x.append(team_rolling_avg[f'Team_{stat}_avg'] - opp_rolling_avg[f'Team_{stat}_avg'])
        x = np.array(x)
        x = np.reshape(x,(-1,1))
        # project x onto vhat
        alpha = np.real(x.T.dot(self.vhat))
        # probability of winning
        return self.log.predict_proba(alpha)[0]
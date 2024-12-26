from src.bracket.AbstractGame import AbstractGame

class BaseGame(AbstractGame):

    def __init__(self, model, round, team1, team2):
        super().__init__(model, round)
        self.__team1 = team1
        self.__team2 = team2

    def get_probs(self):
        self._check_has_parent()
        if self._cached_probs is not None:
            return self._cached_probs

        team_1_prediction = self._model.predict(self.__team1, self.__team2, 'N')[1]
        team_2_prediction = 1 - team_1_prediction
        self._cached_probs = self._sort_dict({self.__team1: team_1_prediction, self.__team2: team_2_prediction})
        return self._cached_probs

    def get_games(self):
        return [self]

    def get_teams(self):
        return [self.__team1, self.__team2]
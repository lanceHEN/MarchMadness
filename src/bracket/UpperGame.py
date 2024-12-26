import numpy as np
from src.bracket.AbstractGame import AbstractGame

class UpperGame(AbstractGame):

    def __init__(self, model, round, game1, game2):
        super().__init__(model, round)
        self.__game1 = game1
        self.__game2 = game2
        self.__game1._add_parent(self)
        self.__game2._add_parent(self)

    def get_probs(self):
        self._check_has_parent()
        if self._cached_probs is not None:
            return self._cached_probs

        probs_map = dict.fromkeys(self.get_teams())
        game_1_probs = self.__game1.get_probs()
        game_2_probs = self.__game2.get_probs()
        # iterate over game 1
        for team in self.__game1.get_teams():
            sum = 0
            for opp in self.__game2.get_teams():
                sum += (game_1_probs[team]
                        * game_2_probs[opp]
                        * self._model.predict(team, opp, 'N'))[1]
            probs_map[team] = sum
        # iterate over game 2
        for team in self.__game2.get_teams():
            sum = 0
            for opp in self.__game1.get_teams():
                sum += (game_2_probs[team]
                        * game_1_probs[opp]
                        * self._model.predict(team, opp, 'N'))[1]
            probs_map[team] = sum

        probs_map = self._sort_dict(probs_map)
        self._cached_probs = probs_map
        return probs_map

    def get_games(self):
        return self.__game1.get_games() + [self] + self.__game2.get_games()

    def get_teams(self):
        return self.__game1.get_teams() + self.__game2.get_teams()
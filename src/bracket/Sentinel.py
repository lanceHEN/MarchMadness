import numpy as np
from src.bracket.AbstractGame import AbstractGame
from src.bracket.BaseGame import BaseGame
from src.bracket.UpperGame import UpperGame

class Sentinel(AbstractGame):

    def __init__(self, model, round, game):
        super().__init__(model, round)
        self.__game = game
        self.__game._add_parent(self)

    def _add_parent(self, parent):
       raise Exception("Cannot add parent to sentinel")

    def get_probs(self):
        raise Exception("Cannot calculate probs for sentinel")

    def get_expected_values(self):
        values_dict = dict.fromkeys(self.get_teams(), 0)
        return values_dict

    def get_games(self):
        return self.__game.get_games()

    def get_teams(self):
        return self.__game.get_teams()
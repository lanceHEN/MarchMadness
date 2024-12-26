from abc import ABC, abstractmethod

'''
The Abstra
'''
class AbstractGame(ABC):

    def __init__(self, model, round):
        self._model = model
        self._round = round
        self._parent = None
        self._cached_probs = None
        self._cached_expected_vals = None

    def _add_parent(self, parent):
        self._parent = parent

    @abstractmethod
    def get_probs(self):
        pass

    def get_expected_values(self):
        self._check_has_parent()
        if self._cached_expected_vals is not None:
            return self._cached_expected_vals

        probs_map = self.get_probs()
        for team in probs_map:
            probs_map[team] = probs_map[team] * team.get_seed() * 2**(self._round - 1) + self._parent.get_expected_values()[team]

        probs_map = self._sort_dict(probs_map)
        self._cached_expected_vals = probs_map
        return probs_map

    def _check_has_parent(self):
        if self._parent is None:
            raise Exception("Parent is not defined")

    @abstractmethod
    def get_games(self):
        pass

    @abstractmethod
    def get_teams(self):
        pass

    def get_round(self):
        return self._round

    @staticmethod
    def _sort_dict(dictionary):
        dict_sorted = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
        return dict_sorted
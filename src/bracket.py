from abc import ABC, abstractmethod
from src.model import Model

"""
The bracket module contains classes that make up a tree-like bracket of the March Madness tournament.

The chief purpose of such a tree is to calculate probabilities of different teams making it to different rounds in the
tournament, as well as their corresponding expected values.

Typical usage example:

# define teams
auburn = Team("Auburn", 1)
lamar = Team("Lamar", 16)
unc = Team("North Carolina", 8)
wisconsin = Team("Wisconsin", 9)

# define and train model
model = MMLogisticRegression('http://barttorvik.com/getgamestats.php?year=2025&csv=1')
model.train()

# define games
basegame_1 = BaseGame(model, auburn, lamar)
basegame_2 = BaseGame(model, unc, wisconsin)

championship = UpperGame(model, 2, basegame_1, basegame_2)

# add sentinel
sentinel = Sentinel(championship)

# get probabilities for winning championship
probs = championship.get_probabilities()

# get expected values for winning basegame_1
basegame_1_expected_vals = basegame_1.get_expected_values()

# get expected value for auburn
auburn_ev = basegame_1_expected_vals[auburn]

"""

class Team:
    """
    A Team is a particular school in the March Madness tournament. All teams have a name and a seed. Equality and hash
    codes for teams are computed according to their name and seed.

    Attributes:
        __name: String representing the team name.
        __seed: Integer representing the team seed.
    """

    def __init__(self, name: str, seed: int):
        """
        Constructs a Team object with the given name and seed.

        Args:
            name (str): The name of the Team.
            seed (int): The seed of the Team (between 1 and 16).

        Raises:
            ValueError: If the given seed is not between 1 and 16.
        """
        if seed < 1 or seed > 16:
            raise ValueError("Seed must be between 1 and 16")
        self.__name = name
        self.__seed = seed

    @property
    def get_name(self) -> str:
        """
        Produces the name of the Team as a string.

        Returns:
            str: The name of the Team.
        """
        return self.__name

    @property
    def get_seed(self) -> int:
        """
        Produces the seed of the Team as an integer.

        Returns:
            int: The seed of the Team.
        """
        return self.__seed

    def __eq__(self, other: object) -> bool:
        """
        Determines if this Team equals the given object. Two Teams are equal iff they have the same name and seed.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: if other is also a Team and has the same name and seed as this Team.
        """
        if not isinstance(other, Team):
            return False
        return self.__name == other.get_name and self.__seed == other.get_seed

    def __hash__(self) -> int:
        """
        Produces a hash value for this Team, using its name and seed.

        Returns:
            int: The hash value of the Team.
        """
        return hash((self.__name, self.__seed))

class Game(ABC):
    """
    A particular matchup in the March Madness tournament.

    A canonical game may either contain two teams (hence representing a "base game") or two
    more Games (hence representing an "upper game" in the later rounds of the tournament). Games can be interpreted
    as nodes of a binary tree, whose root is the championship, and whose leaves are the "base games". All canonical
    games have a Model object to determine win probabilities, and a round they are in. Importantly, all canonical games
    must also have a parent, including the championship. Having a parent for the championship is accomplished by
    instantiating a Sentinel whose only child is the championship node.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the game.
        _round: An integer representing the round of the game.
        _parent: A Game object that is the parent of this Game object.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
        _cached_expected_vals: A dict object that is the output of self.get_expected_vals(), useful for preventing
        duplicate computations.
    """

    def __init__(self, model: Model, round: int):
        """
        Constructs a Game object with the give model and round.

        Args:
            model (Model): A Model object responsible for calculating win probabilities for the game.
            round (int): The round of the game.

        Raises:
            ValueError: If round is not a positive integer.
        """
        if round <= 0:
            raise ValueError('round must be a positive integer')
        self._model = model
        self._round = round
        self._parent = None
        self._cached_probs = None
        self._cached_expected_vals = None

    def _add_parent(self, parent: 'Game') -> None:
        """
        Adds the given Game object as the parent to this Game object.

        Args:
            parent (Game): The requested parent of this Game object.

        Raises:
            RuntimeError: If this Game object already has a parent.
            TypeError: If parent is a BaseGame object.
        """
        if self._parent is not None:
            raise RuntimeError('Game already has a parent')
        if isinstance(parent, BaseGame):
            raise TypeError('parent cannot be a BaseGame object')
        self._parent = parent

    @abstractmethod
    def get_probs(self) -> dict[Team, float]:
        """
        Determines the probability of each possible team in this Game winning this game and any previous games, producing
        a dict from such teams to their probabilities.

        Returns:
            dict: A dict from each possible team in this game to their probability of winning this game and all games in
            earlier rounds.

        Raises:
            RuntimeError: If this Game object does not have a parent.
        """
        pass

    def get_expected_values(self) -> dict[Team, float]:
        """
        Determines the expected points for each possible team in this Game and any later games in higher rounds, producing a dict
        from teams to their expected points. Expected points for a team are calculated by multiplying the probability of
        winning the game by the number of points they would earn, where such points are determined by multiplying the seed
        of the team by 2^round. To account for later expected winnings, expected points in later rounds are added to the
        totals.

        Returns:
            dict: A dict from each possible team in this game to their expected points in this game's round and all later
            rounds.

        Raises:
            RuntimeError: If this Game object does not have a parent.
        """
        self._check_has_parent()
        if self._cached_expected_vals is not None:
            return self._cached_expected_vals

        probs_map = self.get_probs()
        for team in probs_map:
            probs_map[team] = probs_map[team] * team.get_seed * 2**(self._round - 1) + self._parent.get_expected_values()[team]

        probs_map = self._sort_dict(probs_map)
        self._cached_expected_vals = probs_map
        return probs_map

    def _check_has_parent(self) -> None:
        """
        Determines if this Game object does not have a parent, throwing a RuntimeError if so.

        Raises:
            RuntimeError: If this Game object does not have a parent.
        """
        if self._parent is None:
            raise RuntimeError("Parent is not defined")

    @abstractmethod
    def get_games(self) -> dict[int, list['Game']]:
        """
        Produces a dict from round to its corresponding Games, for each Game contained in this Game object
         (including itself).

        Returns:
            dict: a dict from round to its corresponding Games, for each Game contained in this Game object
            (including itself).
        """
        pass

    @abstractmethod
    def get_teams(self) -> list[Team]:
        """
        Produces a list of all teams contained in this Game. The list is produced in order, with the leftmost team first
        and the rightmost team last, where leftmost corresponds to the first team in the base game and rightmost corrsponds
        to the second team in the base game.

        Returns:
            list: a list of all teams contained in this Game.
        """
        pass

    @property
    def get_round(self) -> int:
        """
        Produces the round of this Game.

        Returns:
            int: The round of this Game.
        """
        return self._round

    @staticmethod
    def _sort_dict(dictionary: dict) -> dict:
        """
        Given a dict, produces a sorted version in ascending order.

        Args:
            dictionary (dict): A dict to sort.

        Returns:
            dict: A sorted version of dictionary in ascending order.

        Raises:
            TypeError: If dictionary is not a dict.
        """
        if not isinstance(dictionary, dict):
            raise TypeError('dictionary is not a dict')
        dict_sorted = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
        return dict_sorted

class BaseGame(Game):
    """
    The BaseGame class is a Game in the first part of March Madness, having just two teams. It can be interpreted
    as a leaf of a binary tree, where the tree is the bracket as a whole.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the BaseGame.
        _round: An integer representing the round of the BaseGame.
        _parent: A Game object that is the parent of this BaseGame object.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
        _cached_expected_vals: A dict object that is the output of self.get_expected_vals(), useful for preventing
        duplicate computations.
        __team1: The first team in this BaseGame.
        __team2: The second team in this BaseGame.
    """

    def __init__(self, model: Model, round: int, team1: Team, team2: Team):
        """
        Constructs a BaseGame object with the given model, round, first team, and second team. Note that round may be
        specified for brackets that are not "full", but have unique shapes.

        Args:
            model (Model): A Model object responsible for calculating win probabilities for this BaseGame.
            team1 (Team): The first team of this BaseGame
            team2 (Team): The second team of this BaseGame.

        Raises:
            ValueError if round <= 0
        """
        super().__init__(model, round)
        self.__team1 = team1
        self.__team2 = team2

    def get_probs(self) -> dict[Team, float]:
        self._check_has_parent()
        if self._cached_probs is not None:
            return self._cached_probs

        team_1_prediction = self._model.predict(self.__team1.get_name, self.__team2.get_name, 'N')[1]
        team_2_prediction = 1 - team_1_prediction
        self._cached_probs = self._sort_dict({self.__team1: team_1_prediction, self.__team2: team_2_prediction})
        return self._cached_probs

    def get_games(self) -> dict[int, Game]:
        return {self._round: [self]}

    def get_teams(self) -> list[Team]:
        return [self.__team1, self.__team2]

class UpperGame(Game):
    """
    The UpperGame class is a Game in any later round of March Madness, having 2^round possible teams, which stem from
    the two Game objects it contains. It can be interpreted as a non-leaf node of a binary tree, where the tree is the
    bracket as a whole.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the UpperGame.
        _round: An integer representing the round of the UpperGame.
        _parent: A Game object that is the parent of this UpperGame object.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
        _cached_expected_vals: A dict object that is the output of self.get_expected_vals(), useful for preventing
        duplicate computations.
        __game1: The first game in this UpperGame.
        __game2: The second game in this UpperGame.
    """

    def __init__(self, model: Model, round: int, game1: Game, game2: Game):
        """
        Constructs an UpperGame object with the given model, first game, and second game.

        Args:
        model (Model): A Model object responsible for calculating win probabilities for this UpperGame.
        game1 (Team): The first game of this UpperGame
        game2 (Team): The second game of this UpperGame.

        Raises:
        RuntimeError: If round <= 0
        """
        super().__init__(model, round)
        self.__game1 = game1
        self.__game2 = game2
        self.__game1._add_parent(self)
        self.__game2._add_parent(self)

    def get_probs(self) -> dict[Team, float]:
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
                        * self._model.predict(team.get_name, opp.get_name, 'N'))[1]
            probs_map[team] = sum
        # iterate over game 2
        for team in self.__game2.get_teams():
            sum = 0
            for opp in self.__game1.get_teams():
                sum += (game_2_probs[team]
                        * game_1_probs[opp]
                        * self._model.predict(team.get_name, opp.get_name, 'N'))[1]
            probs_map[team] = sum

        probs_map = self._sort_dict(probs_map)
        self._cached_probs = probs_map
        return probs_map

    def get_games(self) -> dict[int, Game]:
        games = self.__game1.get_games()
        games2 = self.__game2.get_games()
        for round in games.keys():
            games[round] += games2[round]
        games[self._round] = [self]
        return games

    def get_teams(self) -> list[Team]:
        return self.__game1.get_teams() + self.__game2.get_teams()

class Sentinel(Game):
    """
    A Sentinel is a dummy Game whose only purpose is to act as a parent for the championship game (canonical root of the
    bracket tree).

    Sentinels may not have parents, and they do not have any way to calculate their own probabilities.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the game.
        _round: An integer representing the round of the game.
        _parent: A Game object that is the parent of this Game object.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
        _cached_expected_vals: A dict object that is the output of self.get_expected_vals(), useful for preventing
        duplicate computations.
        __game: the championship Game of the tournament.
    """

    def __init__(self, game: Game):
        """
        Constructs a Sentinel object with the given Game.

        Args:
            game (Game): the championship Game (root) of the bracket.
        """
        # NOTE: super deliberately not called because model and round are useless for sentinels and don't need
        # to be initialized
        self.__game = game
        self.__game._add_parent(self)

    # raises a RuntimeError because Sentinels cannot have parents
    def _add_parent(self, parent: Game) -> None:
        raise RuntimeError("Cannot add parent to sentinel")

    # simply raises a RuntimeError because Sentinels cannot calculate probabilities
    def get_probs(self) -> dict[Team, float]:
        raise RuntimeError("Cannot calculate probs for sentinel")

    # simply produces a dict from each team in the sentinel to 0, which is useful for acting as a stopping point for
    # get_expected_values calculations in canonical games
    def get_expected_values(self) -> dict[Team, float]:
        values_dict = dict.fromkeys(self.get_teams(), 0)
        return values_dict

    # produces all games in this Sentinel, not including this Sentinel.
    def get_games(self) -> dict[int, Game]:
        return self.__game.get_games()

    # produces all teams in this Sentinel.
    def get_teams(self) -> list[Team]:
        return self.__game.get_teams()
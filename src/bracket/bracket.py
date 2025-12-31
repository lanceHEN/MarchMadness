from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Iterable

from model import Model

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
        _name: String representing the team name.
        _seed: Integer representing the team seed.
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
        self._name = name
        self._seed = seed

    @property
    def name(self) -> str:
        """
        Produces the name of the Team as a string.

        Returns:
            str: The name of the Team.
        """
        return self._name

    @property
    def seed(self) -> int:
        """
        Produces the seed of the Team as an integer.

        Returns:
            int: The seed of the Team.
        """
        return self._seed

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
        return self.name == other.name and self.seed == other.seed

    def __hash__(self) -> int:
        """
        Produces a hash value for this Team, using its name and seed.

        Returns:
            int: The hash value of the Team.
        """
        return hash((self._name, self._seed))


class Game(ABC):
    """
    A particular matchup in the March Madness tournament.

    A canonical game may either contain two teams (hence representing a "base game") or two
    more Games (hence representing an "upper game" in the later rounds of the tournament). Games can be interpreted
    as nodes of a binary tree, whose root is the championship, and whose leaves are the "base games". All canonical
    games have a Model object to determine win probabilities, and a round they are in.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the game.
        _round: An integer representing the round of the game.
        _teams: Set of all teams within the game.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
            Note because of the tree structure, each game should only have get_probs called once in the recursive tree, so
            the caching is less for expediting the initial call and more for convenience in case it happens to be called again
            (e.g. in get_expected_values).
        _cached_optimal_total_evs: A dict that is the cached output of get_opt_total_evs(), useful for preventing duplicate
            computations. Similar to _cached_probs, get_opt_total_evs() is only called once for each node in the recursive tree
            so this is more for saving time in case it happens to be called again.
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
            raise ValueError("round must be a positive integer")
        self._model = model
        self._round = round
        self._initialize_teams()
        self._cached_probs = None
        self._cached_optimal_total_evs = None

    @abstractmethod
    def get_probs(self) -> Dict[Team, float]:
        """
        Determines the probability of each possible team in this Game winning this game and any previous games, producing
        a dict from such teams to their probabilities.

        EFFECT: Caches result to self._cached_probs.

        Note because of the tree structure, each game should only have get_probs called once in the recursive tree, so
        the caching is less for expediting the initial call and more for convenience in case it happens to be called again
        (e.g. in get_expected_values).

        Returns:
            Dict[Team, float]: A dict from each possible team in this game to their probability of winning this game and all games in
            earlier rounds.
        """
        pass

    def get_expected_values(self) -> Dict[Team, float]:
        """
        Determines the expected points for each possible team in this Game, producing a dict
        from teams to their expected points. Expected points for a team are calculated by multiplying the probability of
        winning the game by the number of points they would earn, where such points are determined by multiplying the seed
        of the team by 2^(round-1) where round is 1 indexed.

        Returns:
            Dict[Team, float]: A dict from each possible team in this game to their expected points for this game.

        """
        evs_map = self.get_probs().copy()
        for team in evs_map:
            evs_map[team] = evs_map[team] * Game._get_value(team, self._round)

        return evs_map

    @abstractmethod
    def get_opt_total_evs(self) -> Dict[Team, Tuple[float, Team]]:
        """
        For each team in this game, finds the optimal total expected value for the bracket subtree
        rooted at this game in the case that team wins this game (and therefore qualifies for it).
        Also returns the losing team for reference (None if it's a base game).

        This is a dynamic programming approach to allow one to find a globally optimal valid
        bracket that maximizes expected values. The bracket can be reconstructed by picking
        the team with the highest total e.v. for the championship, and reconstructing the bracket
        using get_opt_total_evs, called on that team and the losing team, in the child games.

        Returns:
            Dict[Team, Tuple[float, Team]]: Mapping from each team to their optimal total expected
                points, and the losing team in the optimal configuration.
        """
        pass

    @abstractmethod
    def get_games(self) -> Dict[int, List["Game"]]:
        """
        Produces a dict from round to its corresponding Games, for each Game contained in this Game object
         (including itself).

        Returns:
            Dict[int, List['Game']]: a dict from round to its corresponding Games, for each Game contained in this Game object
            (including itself).
        """
        pass

    @property
    def get_teams(self) -> Set[Team]:
        """
        Produces a set of all teams contained in this Game.
        """
        return self._teams

    @abstractmethod
    def _initialize_teams(self) -> None:
        """
        EFFECT: Initializes self._teams
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
    def _get_value(team: Team, round: int):
        """Produces the value for the team winning the given round, given by seed*2^(round-1)."""
        return team.seed * 2 ** (round - 1)


class BaseGame(Game):
    """
    The BaseGame class is a Game in the first part of March Madness, having just two teams. It can be interpreted
    as a leaf of a binary tree, where the tree is the bracket as a whole.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the BaseGame.
        _round: An integer representing the round of the BaseGame.
        _teams: Set of all teams within the game.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
            Note because of the tree structure, each game should only have get_probs called once in the recursive tree, so
            the caching is less for expediting the initial call and more for convenience in case it happens to be called again
            (e.g. in get_expected_values).
        _cached_optimal_total_evs: A dict that is the cached output of get_opt_total_evs(), useful for preventing duplicate
            computations. Similar to _cached_probs, get_opt_total_evs() is only called once for each node in the recursive tree
            so this is more for saving time in case it happens to be called again.
        _team1: The first team in this BaseGame.
        _team2: The second team in this BaseGame.
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
        self._team1 = team1
        self._team2 = team2

        # Have to call this AFTER team1 and team2 are initialized or else an error will raise when trying to initialize
        # self._teams
        super().__init__(model, round)

    def get_probs(self) -> Dict[Team, float]:
        if self._cached_probs is None:
            team_1_prediction = self._model.predict(
                self._team1.name, self._team2.name, "N"
            )[1]
            team_2_prediction = 1 - team_1_prediction
            self._cached_probs = {
                self._team1: team_1_prediction,
                self._team2: team_2_prediction,
            }

        return self._cached_probs

    def get_opt_total_evs(self) -> Dict[Team, Tuple[float, Team]]:
        if self._cached_optimal_total_evs is None:
            probs_map = self.get_probs()
            opt_ev_map = {}
            for team in probs_map:
                opt_ev_map[team] = (
                    self._get_value(team, self._round) * probs_map[team],
                    None,
                )

            self._cached_optimal_total_evs = opt_ev_map

        return self._cached_optimal_total_evs

    def _initialize_teams(self) -> None:
        self._teams = {self._team1, self._team2}

    def get_games(self) -> Dict[int, List[Game]]:
        return {self._round: [self]}


class UpperGame(Game):
    """
    The UpperGame class is a Game in any later round of March Madness, having 2^round possible teams, which stem from
    the two Game objects it contains. It can be interpreted as a non-leaf node of a binary tree, where the tree is the
    bracket as a whole.

    Attributes:
        _model: A Model object responsible for calculating win probabilities for the UpperGame.
        _round: An integer representing the round of the UpperGame.
        _teams: Set of all teams within the game.
        _cached_probs: A dict object that is the output of self.get_probs(), useful for preventing duplicate computations.
            Note because of the tree structure, each game should only have get_probs called once in the recursive tree, so
            the caching is less for expediting the initial call and more for convenience in case it happens to be called again
            (e.g. in get_expected_values).
        _cached_optimal_total_evs: A dict that is the cached output of get_opt_total_evs(), useful for preventing duplicate
            computations. Similar to _cached_probs, get_opt_total_evs() is only called once for each node in the recursive tree
            so this is more for saving time in case it happens to be called again.
        _game1: The first game in this UpperGame.
        _game2: The second game in this UpperGame.
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
        self._game1 = game1
        self._game2 = game2

        # Have to call this AFTER game1 and game2 are initialized or else an error will raise when trying to initialize
        # self._teams
        super().__init__(model, round)

    def get_probs(self) -> Dict[Team, float]:
        if self._cached_probs is None:
            probs_map = dict.fromkeys(self.get_teams)
            game_1_probs = self._game1.get_probs()
            game_2_probs = self._game2.get_probs()
            # iterate over game 1
            for team in game_1_probs:
                sum = 0
                for opp in game_2_probs:
                    sum += (
                        game_1_probs[team]
                        * game_2_probs[opp]
                        * self._model.predict(team.name, opp.name, "N")
                    )[1]
                probs_map[team] = sum
            # iterate over game 2
            for team in game_2_probs:
                sum = 0
                for opp in game_1_probs:
                    sum += (
                        game_2_probs[team]
                        * game_1_probs[opp]
                        * self._model.predict(team.name, opp.name, "N")
                    )[1]
                probs_map[team] = sum

            self._cached_probs = probs_map

        return self._cached_probs

    def get_opt_total_evs(self) -> Dict[Team, Tuple[float, Team]]:
        if self._cached_optimal_total_evs is None:
            probs_map = self.get_probs()
            # Out of all game 1s, which team has the highest value and what team is it?
            g1_opt_evs = self._game1.get_opt_total_evs()
            g1_opt_team = None
            g1_opt_val = 0
            for team in g1_opt_evs:
                if g1_opt_evs[team][0] > g1_opt_val:
                    g1_opt_val = g1_opt_evs[team][0]
                    g1_opt_team = team

            # Same for game 2
            g2_opt_evs = self._game2.get_opt_total_evs()
            g2_opt_team = None
            g2_opt_val = 0
            for team in g2_opt_evs:
                if g2_opt_evs[team][0] > g2_opt_val:
                    g2_opt_val = g2_opt_evs[team][0]
                    g2_opt_team = team

            opt_ev_map = {}
            # Over game 1
            for team in g1_opt_evs:
                # Value for winning this game plus values for each team up to this game
                opt_ev_map[team] = (
                    self._get_value(team, self._round) * probs_map[team]
                    + g1_opt_evs[team][0]
                    + g2_opt_val,
                    g2_opt_team,
                )

            # Over game 2
            for team in g2_opt_evs:
                opt_ev_map[team] = (
                    self._get_value(team, self._round) * probs_map[team]
                    + g2_opt_evs[team][0]
                    + g1_opt_val,
                    g1_opt_team,
                )

            self._cached_optimal_total_evs = opt_ev_map

        return self._cached_optimal_total_evs

    def get_games(self) -> Dict[int, List[Game]]:
        games = self._game1.get_games()
        games2 = self._game2.get_games()
        for round in games.keys():
            games[round] += games2[round]
        games[self._round] = [self]
        return games

    def _initialize_teams(self) -> None:
        self._teams = self._game1.get_teams | self._game2.get_teams

    def get_child_games(self) -> Tuple[Team]:
        """Returns the first and second game in this UpperGame."""
        return self._game1, self._game2


def make_bracket_from_teams(model: Model, teams: Iterable[Tuple[str, int]]) -> Game:
    """
    Given a model and iterable of 2^k teams, where k is a positive number, produces
    the root (championship game) of a symmetric bracket. The bracket is made
    such that every 2 team in teams, left to right, are paired against each
    other in the first round, then in any subsequent round each 2 winners
    from the previous round, left to right, are paired against each other.

    Args:
        model (Model): Model to get probability calculations from.
        teams (Iterable[Tuple[str, int]]): List of teams for the bracket.

    Returns:
        Game: Root node of the bracket, i.e. the championship game.
    """
    # Convert to list of Teams
    teams = [Team(name, seed) for name, seed in teams]

    # Set up first round
    prev_round_games = [
        BaseGame(Model, 1, teams[i], teams[i + 1]) for i in range(len(teams))
    ]

    # Set up other rounds
    round = 2
    while len(prev_round_games) > 1:
        prev_round_games = [
            UpperGame(Model, round, prev_round_games[i], prev_round_games[i + 1])
            for i in range(len(prev_round_games))
        ]
        round += 1

    # Get reference to championship game
    championship = prev_round_games[0]
    return championship

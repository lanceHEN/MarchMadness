import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import uniform
from tqdm import tqdm

from bracket import Game, Team


class BracketLocalSearcher:
    """
    Given all games within the tournament, this class will find a valid bracket, i.e. a mapping of each
    round to list of winners for that round, where each winner in a later round must be a winner in
    all previous rounds, that maximizes total expected value over all games. Local search adjustments
    are made by randomly picking a game, and picking a different winner where teams with higher
    expected values are more likely to be picked. If the old winner also won more games, the
    replacement logic will be recursively applied until the bracket is valid.

    Attributes:
        games_by_round: Mapping from each round to the games within it.
        winners: Mapping from each round to the winners of each game within it.
        total_expected_values: Total expected value for winners.
    """

    def __init__(self, games_by_round: Dict[int, List[Game]]):
        """
        Initializes a BracketLocalSearcher object with the given bracket/mapping of round to game.

        Args:
            games_by_round (Dict[int, List[Game]]): Mapping from each round to the games within it.
        """
        self.games_by_round = games_by_round
        # Randomly pick one for local search - note this ensures all games have equal chances, rather than randomly picking round -> game in round
        self.__games_by_round_indices = [
            (rnd, idx)
            for rnd in games_by_round.keys()
            for idx in range(len(games_by_round[rnd]))
        ]
        self.rounds = set(self.games_by_round.keys())
        self.winners, self.total_expected_values = self.__get_greedy_bracket()

    def __get_greedy_bracket(self) -> Tuple[Dict[int, List[Team]], float]:
        """
        This will, starting from the first round, pick the team with the highest expected value
         as the winner.

        Note this isn't necessarily optimal, as a 16-seeded team might have a higher expected value
         for the first round than the 1-seed but a much lower expected value later on.

        Returns:
            Tuple[Dict[int, List[Team]], float]: Mapping from each round to a list of winners for it,
            along with the total expected values.
        """

        winners = {}
        total_evs = 0
        for round in self.games_by_round.keys():
            winners[round] = []
            prev_winners = set(winners[round - 1]) if round != 1 else {}
            for game in self.games_by_round[round]:
                # Find max expected val team for round, s.t. they also won the last round
                evs = game.get_expected_values()
                # print(evs)
                teams_sorted_evs = list(evs.keys())
                random.shuffle(teams_sorted_evs)
                i = -1
                while round != 1 and (teams_sorted_evs[i] not in prev_winners):
                    i -= 1
                total_evs += evs[teams_sorted_evs[i]]
                winners[round].append(teams_sorted_evs[i])

        return winners, total_evs

    def find_optimal_bracket(
        self,
        num_restarts: int = 1,
        annealing_iterations: int = 1000,
        temperature: float = 30,
        decay_rate: float = 0.999995,
    ) -> None:
        """
        This will run simulated annealing to find a bracket maximizing total expected values, for the given number of iterations, restarting
        from the initial state the given number of times.

        It will update the winners and total_expected_values with the results of the search.

        This function can be ran an arbitary number of times, it will just update the attributes each time.

        Args:
            num_restarts (int): Number of times to restart simulated annealing from the initial state.
            annealing_iterations (int): Number of times to run one restart of simulated annealing.
            temperature (float): The temperature to control how often to accept less valuable bracket states. Should be a positive number.
            decay_rate (float): The rate at which the temperature should exponentially decay. Should be between 0 and 1 inclusive.

        Raises:
            ValueError: If temperature <= 0 or decay_rate < 0 or decay_rate > 1.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive!")
        if decay_rate < 0 or decay_rate > 1:
            raise ValueError("Decay rate must be between 0 and 1!")

        # Don't update self.winners or self.total_expected_values until the very end, and in the meantime make copies.

        best_winners = self.winners
        best_total_expected_values = (
            self.total_expected_values
        )  # Maximize (or in our case minimize negative for simulated annealing)

        for _ in tqdm(
            range(num_restarts), desc="Running restarted simulated annealing..."
        ):
            winners, total_evs = self.__get_greedy_bracket()
            winners, total_evs = self.__simulated_annealing(
                winners, total_evs, annealing_iterations, temperature, decay_rate
            )
            if total_evs > best_total_expected_values:
                best_winners = winners
                best_total_expected_values = total_evs

        self.winners = best_winners
        self.total_expected_values = best_total_expected_values

    def __simulated_annealing(
        self,
        initial_winners: Dict[int, List[Team]],
        initial_total_evs: float,
        annealing_iterations: int = 1000,
        temperature: float = 30,
        decay_rate: float = 0.999995,
    ) -> Tuple[Dict[int, List[Team]], float]:
        """Runs simulated annealing the given number of iterations, with the given temperature and decay rate, returning
        the found bracket and total expected values.

        Local search adjustments are made by randomly picking a game, and picking a different winner where teams with higher
        expected values are more likely to be picked. If the old winner also won more games, the replacement logic will be
        recursively applied until the bracket is valid.

        Args:
            annealing_iterations (int): Number of times to run one restart of simulated annealing.
            temperature (float): The temperature to control how often to accept less valuable bracket states. Should be a positive number.
            decay_rate (float): The rate at which the temperature should exponentially decay. Should be between 0 and 1 inclusive.

        Returns:
            Tuple[Dict[int, List[Team]], float]: The bracket and total expected values found via simulated annealing.
        """

        # Make copy of initial

        best_state = {
            round: initial_winners[round].copy() for round in initial_winners.keys()
        }  # Same team references which is fine
        best_total_evs = initial_total_evs
        current_state = best_state
        current_total_evs = best_total_evs

        for _ in range(annealing_iterations):
            # Fetch a neighboring state
            neighbor_state, neighbor_total_evs = self.__fetch_neighboring_state(
                current_state, current_total_evs
            )

            if neighbor_total_evs < current_total_evs:
                current_state = neighbor_state
                current_total_evs = neighbor_total_evs
                if self.__energy_function(current_total_evs) < self.__energy_function(
                    best_total_evs
                ):
                    best_state = current_state
                    best_total_evs = current_total_evs

            else:
                # print(self.__energy_function(neighbor_total_evs))
                # print(self.__energy_function(current_total_evs))
                # print(temperature)
                accept_prob = np.exp(
                    -(
                        self.__energy_function(neighbor_total_evs)
                        - self.__energy_function(current_total_evs)
                    )
                    / temperature
                )
                if uniform.rvs() < accept_prob:
                    # print("Moved to worse state")
                    current_state = neighbor_state
                    current_total_evs = neighbor_total_evs
                    # No need to update best since we know this is a worse state

            temperature *= decay_rate

        return best_state, best_total_evs

    def __fetch_neighboring_state(
        self, current_state: Dict[int, List[Team]], current_total_evs: float
    ) -> Tuple[Dict[int, List[Team]], float]:
        """
        Given the current state and total expected values, fetches a neighboring state. This is done
        by randomly picking a game, and picking a different winner where teams with higher expected values
        are more likely to be picked. If the old winner also won more games, the replacement logic will be
        recursively applied until the bracket is valid.

        Args:
            current_state (Dict[int, List[Team]]): Current state to make a move from.
            current_total_evs (float): Total expected values for the current state.

        Returns:
            Tuple[Dict[int, List[Team]], float]: The neighboring state and its total expected values.
        """
        neighbor_state = {
            round: current_state[round].copy() for round in current_state.keys()
        }  # Same team references which is fine
        neighbor_total_evs = current_total_evs

        # Pick a random game and find the current winner for it.
        round, game_idx = random.choice(self.__games_by_round_indices)

        cur_winner = current_state[round][game_idx]

        # For this game and any parents where cur_winner is the winner, replace with random with probability proportional to expected val.
        while round in self.rounds and neighbor_state[round][game_idx] == cur_winner:

            game = self.games_by_round[round][game_idx]
            evs = game.get_expected_values()

            if round == 1:
                teams = game.get_teams()
                new_winner = teams[0] if teams[0] != cur_winner else teams[1]
            else:
                possible_winners = []
                possible_winner_evs = []
                first_possible = neighbor_state[round - 1][2 * game_idx]
                if first_possible != cur_winner:
                    possible_winners.append(first_possible)
                    possible_winner_evs.append(evs[first_possible])
                second_possible = neighbor_state[round - 1][2 * game_idx + 1]
                if second_possible != cur_winner:
                    possible_winners.append(second_possible)
                    possible_winner_evs.append(evs[second_possible])

                if len(possible_winners) == 1:
                    new_winner = possible_winners[0]
                else:
                    # Sample by treating expected values as logits
                    possible_winner_probs = softmax(possible_winner_evs)
                    # print(possible_winners)
                    # print(evs[first_possible])
                    # print(possible_winner_evs)
                    # print(type(possible_winner_probs))
                    new_winner = np.random.choice(
                        possible_winners, p=possible_winner_probs
                    )

            neighbor_total_evs -= evs[cur_winner]
            neighbor_total_evs += evs[new_winner]
            neighbor_state[round][game_idx] = new_winner

            round += 1
            game_idx = game_idx // 2

        return neighbor_state, neighbor_total_evs

    @staticmethod
    def __energy_function(total_expected_value: float) -> float:
        """Energy function for simulated annealing: negative of total expected value for minimization."""
        return -total_expected_value

from typing import Dict, List, Tuple

from bracket import Game, Team, UpperGame

"""This module provides functions for finding an optimal March Madness bracket.

For example:

# Assume you already have a reference to the championship game node.

# Find optimal bracket!
opt_bracket, total = find_max_bracket(championship)
for round in sorted(opt_bracket.keys()):
    print(f"Round {round} winners: {[t.name for t in opt_bracket[round]]}")
"""


def find_max_bracket(championship: Game) -> Tuple[Dict[int, List[Team]], float]:
    """Starting from the championship, finds a bracket, or mapping from each round to the teams who won,
    that optimizes total expected points, as well as the total expected points for the bracket.

    Each list in the outputted mapping is ordered from left to right.

    Args:
        championship (Game): The championship game.

    Returns:
        Tuple[Dict[int, List[Team]], float]: Optimal bracket and objective value.
    """
    mapping = {r: [] for r in range(1, championship.round + 1)}

    root_opt_evs = championship.get_opt_total_evs()
    max_team = None
    max_opt_ev = 0
    for team in root_opt_evs:
        if root_opt_evs[team][0] > max_opt_ev:
            max_opt_ev = root_opt_evs[team][0]
            max_team = team

    def rebuild_bracket(game, winning_team):
        """Recursively builds optimal bracket from root."""
        round = game.round
        mapping[round].append(winning_team)

        opt_evs = game.get_opt_total_evs()
        losing_team = opt_evs[winning_team][1]
        if isinstance(game, UpperGame):
            game1, game2 = game.get_child_games()
            # How do we know which game the winner and loser belongs to?
            if winning_team in game1.teams:
                rebuild_bracket(game1, winning_team)
                rebuild_bracket(game2, losing_team)
            else:  # Always game1 first, game2 second
                rebuild_bracket(game1, losing_team)
                rebuild_bracket(game2, winning_team)

    rebuild_bracket(championship, max_team)
    return mapping, max_opt_ev

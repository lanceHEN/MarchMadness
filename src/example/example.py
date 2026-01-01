from bracket import make_team_list, make_bracket_from_teams
from model import LogisticRegression
from optimize import find_max_bracket

"""This file contains an example of setting up a model, measuring its
performance, getting probabilities, and finding an optimal 2025 March Madness bracket."""


def main():
    # Construct model
    model = LogisticRegression(2025, lr=0.0001)

    # Train the model
    model.train(epochs=10000)

    # Example model prediction
    print(model.predict("Auburn", "Oklahoma", "N"))

    # Evaluate accuracy on example season
    print("Accuracy for 2024 season:", model.accuracy(year=2024))

    # Construct 2025 March Madness Bracket

    # Initialize teams
    ALL_TEAMS_RAW = [
        # South
        ("Auburn", 1),
        ("Alabama St.", 16),
        ("Louisville", 8),
        ("Creighton", 9),
        ("Michigan", 5),
        ("UC San Diego", 12),
        ("Texas A&M", 4),
        ("Yale", 13),
        ("Mississippi", 6),
        ("Xavier", 11),
        ("Iowa St.", 3),
        ("Lipscomb", 14),
        ("Marquette", 7),
        ("New Mexico", 10),
        ("Michigan St.", 2),
        ("Bryant", 15),
        # West
        ("Florida", 1),
        ("Norfolk St.", 16),
        ("Connecticut", 8),
        ("Oklahoma", 9),
        ("Memphis", 5),
        ("Colorado St.", 12),
        ("Maryland", 4),
        ("Grand Canyon", 13),
        ("Missouri", 6),
        ("Drake", 11),
        ("Texas Tech", 3),
        ("UNC Wilmington", 14),
        ("Kansas", 7),
        ("Arkansas", 10),
        ("St. John's", 2),
        ("Nebraska Omaha", 15),
        # Midwest
        ("Duke", 1),
        ("American", 16),
        ("Mississippi St.", 8),
        ("Baylor", 9),
        ("Oregon", 5),
        ("Liberty", 12),
        ("Arizona", 4),
        ("Akron", 13),
        ("BYU", 6),
        ("VCU", 11),
        ("Wisconsin", 3),
        ("Montana", 14),
        ("Saint Mary's", 7),
        ("Vanderbilt", 10),
        ("Alabama", 2),
        ("Robert Morris", 15),
        # East
        ("Houston", 1),
        ("SIU Edwardsville", 16),
        ("Gonzaga", 8),
        ("Georgia", 9),
        ("Clemson", 5),
        ("McNeese St.", 12),
        ("Purdue", 4),
        ("High Point", 13),
        ("Illinois", 6),
        ("North Carolina", 11),
        ("Kentucky", 3),
        ("Troy", 14),
        ("UCLA", 7),
        ("Utah St.", 10),
        ("Tennessee", 2),
        ("Wofford", 15),
    ]

    ALL_TEAMS = make_team_list(ALL_TEAMS_RAW)

    championship = make_bracket_from_teams(model, ALL_TEAMS)

    # Find per-team probabilities for winning championship
    champ_probs_map = championship.get_probs()
    for team in ALL_TEAMS:
        print(
            f"Probability of {team.name} winning championship: {100*champ_probs_map[team]}%"
        )

    # Find optimal bracket!
    opt_bracket, total = find_max_bracket(championship)
    for round in sorted(opt_bracket.keys()):
        print(f"Round {round} winners: {[t.name for t in opt_bracket[round]]}")

    print(f"Total expected points: {total}")


if __name__ == "__main__":
    main()

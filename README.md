# March Madness Bracket Predictor

## Overview
This project provides a comprehensive framework for March Madness bracket analysis with two primary objectives:
1. Predicting **game outcome probabilities** using machine learning models trained on historical college basketball data.
2. Finding the **optimal bracket that maximizes expected points** for office pool competitions.

The system uses a tree-based bracket structure combined with machine learning models to simulate tournament outcomes and optimize bracket selections through dynamic programming.

## Project Structure

```
src/
├── bracket/
│   ├── __init__.py
│   └── bracket.py          # Core bracket data structures
├── model/
│   ├── __init__.py
│   └── model.py            # ML models for game prediction
├── preprocess/
│   ├── __init__.py
│   └── preprocess.py       # Data loading and preprocessing
├── optimize/
│   ├── __init__.py
│   └── optimize.py         # Bracket optimization algorithms
└── example/
    └── example.py          # Usage examples
```

### Core Modules

#### `bracket/bracket.py`
Defines the tree-based bracket structure for March Madness tournaments:
- **`Team`**: Represents individual basketball teams with name and seed (1-16).
- **`Game` (Abstract Base Class)**: Represents tournament matchups as nodes in a binary tree.
  - **`BaseGame`**: Leaf nodes representing first-round games between two teams.
  - **`UpperGame`**: Internal nodes representing later-round games between winners of previous games.
- **Probability Calculations**: Recursively computes win probabilities for all teams reaching each round.
- **Expected Value Calculations**: Computes expected points using the formula: `seed × 2^(round-1) × win_probability`.
- **Utility Functions**:
  - `make_team_list()`: Converts team data tuples into Team objects.
  - `make_bracket_from_teams()`: Constructs a symmetric tournament bracket from a list of teams.

#### `model/model.py`
Machine learning models for predicting game outcomes using Barttorvik.com statistics:
- **`Model` (Abstract Base Class)**: Base class with Linear Discriminant Analysis (LDA) projection for dimensionality reduction.
- **`LogisticRegression`**: Binary classifier with L2 regularization to predict win probabilities.
- **`MLP`**: Multi-layer perceptron with configurable hidden layers, ReLU activation, layer normalization, and dropout regularization.
- **Feature Engineering**: 
  - Rolling averages (10-game window) for team statistics.
  - Statistical differentials between opponents.
  - Advanced metrics: adjusted offensive/defensive efficiency, effective field goal percentage, turnover rate, rebounding percentage, free throw rate.
- **Venue Adjustment**: Accounts for home court advantage (H), neutral sites (N), and away games (A).

#### `preprocess/preprocess.py`
Data pipeline for loading and preprocessing college basketball statistics:
- **`get_data(year)`**: Fetches game data from Barttorvik.com for a specified season.
- **Data Processing**:
  - Converts categorical variables (results, venues) to numerical format.
  - Computes rolling averages for team and opponent statistics.
  - Calculates statistical differentials between teams.
  - Applies feature scaling for model training.
- **Output**: Returns pandas DataFrame with raw data plus NumPy arrays (X, y) for model training.

#### `optimize/optimize.py`
Dynamic programming algorithm for finding optimal brackets:
- **`find_max_bracket(championship)`**: Starting from the championship game, finds the bracket configuration that maximizes total expected points.
- **Algorithm**: Uses the `get_opt_total_evs()` method from Game objects to recursively determine optimal team selections at each node.
- **Output**: Returns a mapping from each round to winning teams (ordered left-to-right) plus the total expected value.

#### `example/example.py`
Demonstrates complete workflow for bracket analysis:
1. Model construction and training
2. Accuracy evaluation on historical data
3. Team initialization for 2025 tournament (all four regions)
4. Bracket construction
5. Championship probability calculations
6. Optimal bracket generation

## Installation

### Dependencies
```bash
pip install numpy pandas scikit-learn
```

### Requirements
- Python 3.7+
- Internet connection (for fetching Barttorvik.com data)

## Usage

### Basic Example

```python
from bracket import make_team_list, make_bracket_from_teams
from model import LogisticRegression
from optimize import find_max_bracket

# Define teams (name, seed)
ALL_TEAMS_RAW = [
    ("Auburn", 1),
    ("Alabama St.", 16),
    ("Louisville", 8),
    ("Creighton", 9),
    # ... more teams
]

# Create Team objects
ALL_TEAMS = make_team_list(ALL_TEAMS_RAW)

# Train model
model = LogisticRegression(2025, lr=0.0001)
model.train(epochs=10000)

# Build bracket
championship = make_bracket_from_teams(model, ALL_TEAMS)

# Get championship probabilities
champ_probs = championship.get_probs()
for team in ALL_TEAMS:
    print(f"{team.name}: {100*champ_probs[team]:.2f}%")

# Find optimal bracket
opt_bracket, total_ev = find_max_bracket(championship)
print(f"Total expected points: {total_ev:.2f}")
```

### Model Training and Evaluation

```python
# Train model
model = LogisticRegression(2025, lr=0.0001)
model.train(epochs=10000)

# Evaluate on historical data
accuracy_2024 = model.accuracy(year=2024)
print(f"2024 Season Accuracy: {accuracy_2024:.2%}")

# Make predictions
prob_team1_wins = model.predict("Duke", "North Carolina", "N")[1]
```

### Analyzing Specific Rounds

```python
# Get all games organized by round
games_by_round = championship.get_games()

# Analyze Sweet 16 (Round 3)
for game in games_by_round[3]:
    evs = game.get_expected_values()
    for team, ev in evs.items():
        print(f"{team.name}: {ev:.2f} expected points")
```

## Key Features

- **Tree-Based Bracket Representation**: Efficient recursive structure for probability and expected value calculations.
- **Probabilistic Modeling**: Combines historical statistics with machine learning to predict outcomes.
- **Dynamic Programming Optimization**: Globally optimal bracket selection that maximizes expected points.
- **Caching**: Prevents redundant calculations in the recursive tree structure.
- **Rolling Statistics**: Incorporates recent performance trends (10-game windows).
- **Advanced Metrics**: Uses Barttorvik's adjusted efficiency and Four Factors statistics.
- **Flexible Scoring**: Point values scale with round: `seed × 2^(round-1)`.

## Data Source

This project uses real-time college basketball data from [Bart Torvik's College Basketball Ratings](https://barttorvik.com/), which provides:
- Adjusted offensive and defensive efficiency ratings
- Four Factors statistics (eFG%, TOV%, ORB%, FTR)
- Historical game results dating back to 2008

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

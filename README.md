# March Madness Bracket Predictor

## Overview
This project focuses on two primary objectives:
1. Finding the **most probable March Madness bracket** based on game outcome probabilities.
2. Identifying the **most profitable bracket for office pools**, leveraging probabilistic insights.

## Project Structure
The project consists of the following core components:

### 1. `bracket.py`
This module defines the fundamental classes and structure for representing a March Madness bracket. Key components include:
- **Teams**: Represents individual basketball teams.
- **Games**: Represents individual matchups and their relationships in the tournament bracket.
- **Bracket Management**: Utilities to navigate and simulate games within a bracket structure.

### 2. `madness.py`
This script initializes the tournament setup, models the games, and calculates expected values for various scenarios:
- **Team Initialization**: Sets up teams for each region (South, West, Midwest, East) based on ESPN seeding predictions.
- **Game Simulation**: Uses logistic regression predictions to simulate game outcomes.
- **Bracket Progression**: Progresses through rounds until the championship, calculating expected values at each stage.

### 3. `model.py`
This module defines machine learning models to predict game outcomes based on historical data:
- **Logistic Regression Model**: A binary classifier with L2 regularization to predict the probability of one team defeating another.
- **Feature Engineering**: Computes rolling averages and differentials for team stats to enhance predictive accuracy.
- **Data Pipeline**: Loads and preprocesses data from Barttorvik.com, including adjustments for venue and statistical comparisons.

## Usage
### Dependencies
This project relies on the following Python libraries:
- `numpy`
- `pandas`

Install these dependencies using:
```bash
pip install numpy pandas
```

### Running the Project
1. **Train the Model**:
   Train the logistic regression model using historical game data:
   ```python
   from src/model import LogisticRegression
   model = LogisticRegression(2025)
   model.train()
   ```

2. **Initialize and Simulate Brackets**:
   Use `madness.py` to simulate the tournament and calculate expected values for brackets:
   ```bash
   python madness.py
   ```

3. **View Results**:
   The script outputs the expected values for each matchup, helping identify the most probable and profitable brackets.

## Features
- **Probabilistic Modeling**: Uses logistic regression to compute game outcome probabilities.
- **Dynamic Bracket Simulation**: Progresses through the tournament based on model predictions.
- **Rolling Statistics**: Incorporates advanced stats like adjusted offensive and defensive efficiencies for better accuracy.

## Future Enhancements
- Add more model types (e.g., SVM, Neural Networks) to compare predictive accuracy.
- Include features like player-level statistics or injury data.
- Optimize for different office pool scoring systems.

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality or accuracy of the models.

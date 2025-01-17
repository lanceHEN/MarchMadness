from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from keras.src.backend.jax.nn import sigmoid
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.preprocess import get_data, ven_map, res_map, stats_columns

"""
The model module contains classes useful for predicting the likelihood of one college basketball team beating another
college basketball team, for a given year, at a given venue.

All models are created by calling on their constructor with the appropriate year for team data. They are then trained
on such data via the train() method, and may then make predictions for a given matchup via the predict method.

Example usage:

# init model
log = LogisticRegression(2025)

# train model
log.train()

# make a prediction
predictions = log.predict("Northeastern", "Boston University", "H")

print("Probability of Northeastern winning: ", predictions[1])
print("Probability of BU winning: ", predictions[0])

"""

class Model(ABC):
    """
    A binary classification model capable of producing the probability of one team beating another in March Madness,
    given a particular year of the tournament.

    All models get their data from the barttorvik.com database, using team statistics for a given year. Such stats
    include adjusted offensive efficiency, turnover ratings, free throws ratings, etc., and are computed on a rolling
    average basis, and then subtracted between the two teams for a given game to produce differentials for predictions.
    Such data is then automatically projected onto a one dimensional space (self.alphas) for easier computation.
    Importantly, the model is not trained on construction. Rather, it must have its train method called.
    """

    def __init__(self, year: int, lr: float):
        """
        Constructs a model using barttorvik.com data for the given year.

        Args:
            year (int): The year to draw data from.
            lr (float): Learning rate.

        Raises:
            ValueError: If year is not a positive integer between 2008 and the current year, inclusive.
        """
        self._df, self._X, self._y = get_data(year)
        self._vhat = self._get_vhat(self._X, self._y)
        self._alphas = self._project(self._X)
        self._w = None
        self._lr = lr

    def _get_vhat(self, X, y):
        # perform LDA, projecting onto 1 dimension
        X1 = X[y.flatten()==1]
        X0 = X[y.flatten()==0]
        u = np.reshape(np.mean(X0, axis=0) - np.mean(X1, axis=0), (X.shape[1],1))
        S1 = u.dot(u.T)
        C0 = np.eye(X0.shape[0]) - 1/X0.shape[0] * np.ones((X0.shape[0],X0.shape[0]))
        C1 = np.eye(X1.shape[0]) - 1/X1.shape[0] * np.ones((X1.shape[0],X1.shape[0]))
        S2 = X0.T.dot(C0.dot(X0)) + X1.T.dot(C1.dot(X1))
        Q = np.linalg.inv(S2).dot(S1)
        #print(np.linalg.eig(Q))
        most_important_idx = np.argmax(np.linalg.eig(Q)[0])
        vhat = np.reshape(np.linalg.eig(Q)[1][:,most_important_idx],(-1,1))
        return vhat

    def _project(self, X):
        #project data
        alphas = X.dot(self._vhat)
        alphas = np.real(alphas)
        return alphas

    @abstractmethod
    def train(self, epochs=10000) -> None:
        """
        EFFECT: Trains this model with the data given to it on construction.

        Parameters:
            epochs (int): Number of iterations to train for

        Raises:
            RuntimeError: If this model has already been trained.
        """
        pass

    @abstractmethod
    def accuracy(self, year: int) -> float:
        """
        Produces the percentage of correctly predicted games using data for the given year, as a float between 0 and 1.

        Args:
            year (int): The year to determine accuracy for.

        Returns:
            float: the percentage of correctly predicted games for the given year.

        Raises:
            ValueError: If year is not a positive integer between 2008 and the current year, inclusive.
        """
        pass

    def _compute_rolling_average(self, team: str, window=10) -> dict[str, float]:
        """
        Computes the rolling average stats for a given team from the last 10 games, as a dict from the name of the stat to
        its corresponding value, where each stat is named 'Team_{stat}_avg'.

        Args:
            team (Team): the team to obtain rolling averages for.
            window (int): how many games to compute rolling averages from (set to 10).

        Returns:
            dict: a dict from the name of the stat to
            its corresponding value, where each stat is named 'Team_{stat}_avg'.

        Raises:
            TypeError: If team is not a string or window is not an int.
            ValueError: If team is not recognized or if window is not positive.
        """
        if not isinstance(team, str):
            raise TypeError('team must be a string.')
        if not isinstance(window, int):
            raise TypeError('window must be an integer.')
        team_games = self._df[self._df['Team'] == team]
        if len(team_games) == 0:
            raise ValueError('Team not recognized: ' + str(team))
        if window <= 0:
            raise ValueError('Window must be positive')
        rolling_avg = {}

        for stat in stats_columns:
            rolling_avg[f'Team_{stat}_avg'] = team_games[stat].tail(window).mean()

        return rolling_avg

    def predict(self, team1: str, team2: str, venue: str) -> np.ndarray:
        """
        Produces the probabilities of team1 and team2 winning a matchup at a given venue, as a 1-d numpy array where arr[1]
        is the probability of team1 winning and team2 losing, and arr[0] is the probability of team1 losing and team2
        winning.

        Args:
            team1 (str): the first team in the matchup.
            team2 (str): the second team in the matchup.
            venue (str): the venue of the matchup (must be 'H' for team 1 home, 'N' for neutral, and 'A' for team 1 away).

        Returns:
            np.ndarray: a 1-d numpy array where arr[1]
            is the probability of team1 winning and team2 losing, and arr[0] is the probability of team1 losing and team2
            winning.

        Raises:
            TypeError: If team1 or team2 or venue are not strings.
            ValueError: If team1 or team2 are not recognized, or venue is not one of 'H', 'N', 'A'.
        """
        if venue not in ven_map.keys():
            raise ValueError("venue must be one of the following strings: 'H', 'N', and 'A'.")

        # obtain rolling averages
        team_rolling_avg = self._compute_rolling_average(team1)
        opp_rolling_avg = self._compute_rolling_average(team2)
        # obtain x vector
        x = [ven_map[venue]]
        for stat in stats_columns:
            x.append(team_rolling_avg[f'Team_{stat}_avg'] - opp_rolling_avg[f'Team_{stat}_avg'])
        x = np.array(x)
        x = np.reshape(x,(-1,1))
        # project x onto vhat
        alpha = np.real(x.T.dot(self._vhat))
        # probability of winning and losing
        q1 = self.forward(alpha)
        q0 = 1 - q1
        return np.array([q0, q1])

    @staticmethod
    def _sigmoid(z) -> float:
        return 1/(1 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(z) -> float:
        s = sigmoid(z)
        return s * (1 - s)


class LogisticRegression(Model):
    """
    A logistic regression binary march madness classification model, with an L2 penalty to prevent overfitting.

    Attributes:
        _alphas: one dimensional numpy array of input data for each sample (game).
        _df: pandas dataframe for the raw stats for each matchup in the given year, including individual team stats
         and differentials.
        _res_map: dict mapping the string 'W' and 'L' to 1 and 0, respectively.
        _ven_map: dict mapping 'H', 'N', and 'A' to 1, 0, and -1, respectively, to convert from categorical data.
        _y: one dimensional numpy array of labels for each sample (1: the given team won, 0: they lost).
    """

    def __init__(self, year: int, lr: float):
        """
        Constructs a MMLogisticRegression object with the given year to obtain data from.

        Args:
            year (int): The year to draw data from.
            lr (float): The learning rate of the model.

        Raises:
            TypeError: If year is not an integer
            ValueError: If year is not a positive integer between 2008 and the current year, inclusive.
        """
        super().__init__(year, lr)
        self.__w = None

    def train(self, epochs=10000) -> None:
        if self.__w is not None:
            raise RuntimeError("Model has already been trained!")
        print("Training model, this may take a while...")
        n, d = self._alphas.shape
        phi = np.hstack((self._alphas, np.ones((n, 1))))

        self.__w = np.zeros((d + 1, 1))

        for epoch in range(epochs):
            yhats, phi = self.forward(self._alphas)
            #print('yh',yhats.shape)
            u = self.__bce_grad(self._y, yhats, phi, self.__w)
            self.__w = self.__w - self._lr * u

        print("Model trained!")

    def forward(self, alphas):
        phi = np.hstack((alphas, np.ones((alphas.shape[0],1))))
        q1 = self._sigmoid(phi.dot(self.__w))
        return q1, phi

    def accuracy(self, year):
        other_df, other_X, other_y = get_data(year)
        other_alphas = self._project(other_X)

        correct = 0
        for i, alpha in enumerate(other_alphas):
            alpha = np.reshape(alpha, (1, 1))
            yhat, _ = self.forward(alpha)
            yhat_round = 1 if yhat > 0.5 else 0
            if yhat_round == other_y[i]:
                correct += 1

        return correct / other_alphas.shape[0]

    # binary cross entropy gradient with l2 regularizer
    @staticmethod
    def __bce_grad(y, yhats, phi, w, lambda_=0.001):
        mean = np.mean((yhats - y) * phi, axis=0).T
        mean = np.reshape(mean, (-1,1))
        return mean + lambda_ * 2 * w # add ridge regularizer

class MLP(Model):
    # MLP with one hidden layer. made with numpy exclusively.

    # allow for variable num hidden layers
    def __init__(self, year: int, n_hidden = 1, width_hidden=8, lr=0.00001):
        super().__init__(year, lr)

        # input layer
        self.W = [np.random.randn(self._alphas.shape[1], width_hidden) * 0.01]
        self.b = [np.zeros((1,width_hidden))]
        # hidden layer(s)
        for i in range(n_hidden):
            self.W.append(np.random.randn(width_hidden, width_hidden) * 0.01)
            self.b.append(np.zeros((1,width_hidden)))

        # output layer
        self.W.append(np.random.randn(width_hidden, 1) * 0.01)
        self.b.append(np.zeros((1,1)))

    @staticmethod
    def __bce_grad_simplified(y, yhats):
        return (yhats - y) / (np.maximum((yhats * (1 - yhats)),1e-8))

    @staticmethod
    def __bce(y, yhats):
        return -np.mean(y * np.log(yhats + 1e-8) + (1 - y) * np.log(1 - yhats + 1e-8))

    def train(self, epochs=10000) -> None:
        for epoch in range(epochs):
            yhats= self.forward(self._alphas)
            self.backward(self._y, yhats)
            print(self.__bce(self._y, yhats))

        print("Model trained!")

    @staticmethod
    def __relu(x):
        return np.maximum(0,x)

    @staticmethod
    def __relu_deriv(x):
        return np.where(x > 0, 1, 0)

    def forward(self, alpha: np.ndarray):
        # iteratively activate each layer
        self.activations = [alpha] # need input layer to start from
        self.zs = []
        # for the hidden layers
        for i in range(len(self.W) - 1):
            # dot previous activation with current W and add bias
            z = np.dot(self.activations[-1], self.W[i]) + self.b[i]
            # add aggregates to zs (to be used for backprop)
            self.zs.append(z)
            activation = self.__relu(z)
            self.activations.append(activation)

        # output layer only (uses sigmoid)
        z = np.dot(self.activations[-1], self.W[-1]) + self.b[-1]
        self.zs.append(z)
        yhat = self._sigmoid(z)
        self.activations.append(yhat)

        return yhat

    def backward(self, y, yhats):
        W_grads = [None] * len(self.W)
        b_grads = [None] * len(self.b)

        # get it started with the very last connection, output
        out_err = self.__bce_grad_simplified(y, yhats)
        delta = out_err * self._sigmoid_deriv(self.zs[-1])

        # go backward iteratively to backprop properly
        for i in reversed(range(len(self.W))):
            W_grads[i] = np.dot(self.activations[i].T, delta)
            b_grads[i] = np.sum(delta, axis=0, keepdims=True)

            # if not at the end, calculate next gradient
            if i > 0:
                delta = np.dot(delta, self.W[i].T) * self.__relu_deriv(self.zs[i-1])

        # finally update all weights
        for i in range(len(self.W)):
            self.W[i] -= self._lr * W_grads[i]
            self.b[i] -= self._lr * b_grads[i]

    def accuracy(self, year):
        other_df, other_X, other_y = get_data(year)
        other_alphas = self._project(other_X)

        correct = 0
        for i, alpha in enumerate(other_alphas):
            yhat = 1 if self.forward(alpha.reshape(1, -1)) > 0.5 else 0
            if yhat == other_y[i]:
                correct += 1

        return correct / other_alphas.shape[0]
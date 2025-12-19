from abc import ABC, abstractmethod
import numpy as np
from preprocess import get_data, ven_map, stats_columns

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

    def __init__(self, year: int, lr: float, projected_dim: int=2):
        """
        Constructs a model using barttorvik.com data for the given year.

        Args:
            year (int): The year to draw data from.
            lr (float): Learning rate.

        Raises:
            ValueError: If year is not a positive integer between 2008 and the current year, inclusive,
                or if projected_dim < 1.
        """
        self._df, self._X, self._y = get_data(year)
        self._vhats = self._get_vhats(self._X, self._y, projected_dim)
        self._alphas = self._project(self._X)
        self._w = None
        self._lr = lr

    def _get_vhats(self, X, y, projected_dim):
        # perform LDA, projecting onto 2 dimensions
        X1 = X[y.flatten()==1] # wins
        X0 = X[y.flatten()==0] # losses
        u = np.reshape(np.mean(X0, axis=0) - np.mean(X1, axis=0), (X.shape[1],1))
        S1 = u.dot(u.T)
        C0 = np.eye(X0.shape[0]) - 1/X0.shape[0] * np.ones((X0.shape[0],X0.shape[0]))
        C1 = np.eye(X1.shape[0]) - 1/X1.shape[0] * np.ones((X1.shape[0],X1.shape[0]))
        S2 = X0.T.dot(C0.dot(X0)) + X1.T.dot(C1.dot(X1))
        Q = np.linalg.inv(S2).dot(S1)
        #print(np.linalg.eig(Q))
        most_important_indices = np.argpartition(np.linalg.eig(Q)[0], -projected_dim)[-projected_dim:]
        vhats = np.linalg.eig(Q)[1][:,most_important_indices] # d, projected_dim
        return vhats

    def _project(self, X):
        #print(X.shape)
        #print(self._vhats)
        #note X is expected to have each sample as a row
        alphas = X.dot(self._vhats) # n, projected_dim
        alphas = np.real(alphas)
        return alphas

    @abstractmethod
    def train(self, epochs=10000) -> None:
        """
        EFFECT: Trains this model with the data given to it on construction.

        Args:
            epochs (int): Number of iterations to train for

        Raises:
            RuntimeError: If this model has already been trained.
        """
        pass
    
    @abstractmethod
    def _forward(self, phi: np.ndarray) -> np.ndarray:
        """Given phi (data plus bias column) of shape (n, d+1), produces win probabilities.
        
        Args:
            phi (np.ndarray): Data plus bias column.
            
        Returns:
            np.ndarray: Win probabilities for each sample."""
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
        alphas = self._project(x.T) # 1, projected_dim
        # probability of winning and losing
        phi = np.hstack((alphas, np.ones((alphas.shape[0],1))))
        q1 = self._forward(phi)
        q0 = 1 - q1
        return np.array([q0, q1])

    @staticmethod
    def _sigmoid(z) -> float:
        return 1/(1 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(z) -> float:
        s = Model._sigmoid(z)
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

        for _ in range(epochs):
            yhats = self._forward(phi)
            #print('yh',yhats.shape)
            u = self.__bce_grad(self._y, yhats, phi, self.__w)
            self.__w = self.__w - self._lr * u

        print("Model trained!")

    def _forward(self, phi):
        q1 = self._sigmoid(phi.dot(self.__w))
        return q1

    def accuracy(self, year):
        other_df, other_X, other_y = get_data(year)
        other_alphas = self._project(other_X)

        correct = 0
        for i, alphas in enumerate(other_alphas):
            alphas = np.reshape(alphas, (1, alphas.shape[0]))
            phi = np.hstack((alphas, np.ones((1, 1))))
            yhat = self._forward(phi)
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
    def __init__(self, year: int, n_hidden = 1, width_hidden=8, lr=0.00001, dropout=0.1):
        super().__init__(year, lr)

        # input layer
        self.W = [np.random.normal(0, 1/(self._alphas.shape[1])**2, (self._alphas.shape[1], width_hidden))]
        self.b = [np.zeros((1,width_hidden))]
        self.dropout = dropout
        # hidden layer(s)
        for i in range(n_hidden):
            self.W.append(np.random.normal(0, 1/(n_hidden)**2, (width_hidden, width_hidden)))
            self.b.append(np.ones((1,width_hidden)))

        # output layer
        self.W.append(np.random.normal(0, 1/(n_hidden)**2, (width_hidden, 1)))
        self.b.append(np.ones((1,1)))

    @staticmethod
    def __bce_grad_simplified(y, yhats):
        return (yhats - y) / (np.maximum((yhats * (1 - yhats)),1e-8))

    @staticmethod
    def __bce(y, yhats):
        return -np.mean(y * np.log(yhats + 1e-8) + (1 - y) * np.log(1 - yhats + 1e-8))

    def train(self, epochs=10000, lambda_=1) -> None:
        phi = np.hstack((self._alphas, np.ones((n, 1))))
        for _ in range(epochs):
            yhats= self._forward(phi)
            self.backward(self._y, yhats, lambda_)
            print(self.__bce(self._y, yhats))

        print("Model trained!")

    @staticmethod
    def __layer_norm(x):
        epsilon = 1e-8
        xmean = np.mean(x, axis=1, keepdims=True)
        xvar = np.var(x, axis=1, keepdims=True)
        xhat = (x - xmean) / np.sqrt(xvar + epsilon)
        return xhat

    @staticmethod
    def __relu(x):
        return np.maximum(0,x)

    @staticmethod
    def __relu_deriv(x):
        return np.where(x > 0, 1, 0)

    def _forward(self, phi: np.ndarray, train=True):
        # iteratively activate each layer
        self.activations = [phi] # need input layer to start from
        self.zs = []
        # for the hidden layers
        for i in range(len(self.W) - 1):
            # dot previous activation with current W and add bias
            # z = aw + b
            z = np.dot(self.activations[-1], self.W[i]) + self.b[i]
            # add aggregates to zs (to be used for backprop)
            self.zs.append(z)
            z = self.__layer_norm(z)
            a = self.__relu(z)
            # using dropout technique discussed in AlexNet paper!
            if train:
                # because a contains activations for all neurons, we have to be careful about not setting everything to 0
                drop_mask = np.random.rand(a.shape[0], a.shape[1]) > self.dropout
                a *= drop_mask
            else:
                a /= (1 - self.dropout) # scale during eval to maintain correct expected val
            self.activations.append(a)

        # output layer only (uses sigmoid)
        z = np.dot(self.activations[-1], self.W[-1]) + self.b[-1]
        self.zs.append(z)
        yhat = self._sigmoid(z)

        self.activations.append(yhat)

        return yhat

    def backward(self, y, yhats, lambda_=1):
        W_grads = [None] * len(self.W)
        b_grads = [None] * len(self.b)

        # get it started with the very last connection, output
        # dL/df * df/dz(L) = dL/dz(L)
        out_err = self.__bce_grad_simplified(y, yhats) # dL/df
        delta = out_err * self._sigmoid_deriv(self.zs[-1]) #df/dz(L)

        # compute grads for the output layer
        # dL/dW(L) = delta(L)*a(L-1) # error out * error in
        W_grads[-1] = np.dot(self.activations[-2].T, delta)
        # dL/db(L) = delta(L) nice and easy!
        b_grads[-1] = np.sum(delta, axis=0, keepdims=True)

        # go backward iteratively to backprop properly
        for i in reversed(range(len(self.W) - 1)):
            # delta = (w(l+1)^T*delta(l+1)) * df/dz
            delta = np.dot(delta, self.W[i+1].T) * self.__relu_deriv(self.zs[i-1]) # backprop through relu
            # dL/dw(l) = a(l-1) * delta(l)
            W_grads[i] = np.dot(self.activations[i].T, delta) + lambda_ * 2 * self.W[i] # L2 norm to reduce overfitting
            # dL/db(1) = delta(l)
            b_grads[i] = np.sum(delta, axis=0, keepdims=True)

            # layer norm grad
            delta = self.__layer_norm_backward(delta, self.zs[i-1]) # backprop through layer norm

        # finally update all weights
        for i in range(len(self.W)):
            self.W[i] -= self._lr * W_grads[i]
            self.b[i] -= self._lr * b_grads[i]

    def __layer_norm_backward(self, grad_output, z):
        epsilon = 1e-8
        n, features = z.shape

        # Forward pass components
        mean = np.mean(z, axis=1, keepdims=True)
        var = np.var(z, axis=1, keepdims=True)
        std = np.sqrt(var + epsilon)
        z_norm = (z - mean) / std

        # backprop
        grad_z_norm = grad_output / std  # Gradient w.r.t. normalized z
        grad_variance = np.sum(grad_output * (z - mean) * (-0.5) * (std**-3), axis=1, keepdims=True)
        grad_mean = np.sum(grad_output * (-1 / std), axis=1, keepdims=True) + grad_variance * np.mean(-2 * (z - mean), axis=1, keepdims=True)

        return grad_z_norm + (grad_variance * 2 * (z - mean) / features) + (grad_mean / features)

    def accuracy(self, year):
        other_df, other_X, other_y = get_data(year)
        other_alphas = self._project(other_X)

        correct = 0
        for i, alphas in enumerate(other_alphas):
            phi = phi = np.hstack((alphas, np.ones((n, 1))))
            yhat = 1 if self._forward(phi, train=False) > 0.5 else 0
            if yhat == other_y[i]:
                correct += 1

        return correct / other_alphas.shape[0]
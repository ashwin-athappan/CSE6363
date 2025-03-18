import numpy as np
from decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, weak_learner=DecisionTree, num_learners=10, learning_rate=1.0):
        """
        Parameters:
        weak_learner: DecisionTree class
        num_learners: int -> maximum number of weak learners
        learning_rate: float -> weight applied to each learner
        """
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.learners = []
        self.alphas = []

    def _calculate_error(self, learner, X, y, weights):
        predictions = learner.predict(X)
        return np.sum(weights * (predictions != y)) / np.sum(weights)

    def _calculate_alpha(self, error):
        return 0.5 * self.learning_rate * np.log((1 - error) / max(error, 1e-10))

    def fit(self, X, y):
        """
        Fit AdaBoost classifier where y in {-1, +1}
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.num_learners):
            learner = self.weak_learner()
            learner.fit(X, y)

            error = self._calculate_error(learner, X, y, weights)
            alpha = self._calculate_alpha(error)

            self.learners.append(learner)
            self.alphas.append(alpha)

            predictions = learner.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            if error < 1e-10:
                break

    def predict(self, X):
        """
        Predict using sign of weighted sum of weak learners
        """
        X = np.array(X)
        predictions = np.zeros(X.shape[0])

        for learner, alpha in zip(self.learners, self.alphas):
            predictions += alpha * learner.predict(X)

        return np.sign(predictions)
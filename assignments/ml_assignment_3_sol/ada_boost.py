import numpy as np
from decision_tree import DecisionTree

class AdaBoost:
    """
    AdaBoost Classifier using Decision Trees as weak learners.
    """

    def __init__(self, weak_learner=DecisionTree, num_learners=10, learning_rate=1.0):
        """
        Initialize the AdaBoost model.

        Parameters:
        - weak_learner: The classifier used as a weak learner (default is DecisionTree).
        - num_learners: The maximum number of weak learners to use.
        - learning_rate: The weight applied to each weak learner per iteration.
        """
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.learners = []
        self.alphas = []
        self.label_count = None

    def _calculate_amount_of_say(self, base_learner, X, y):
        """Calculates the amount of say (alpha) for the weak learner."""
        K = self.label_count
        preds = base_learner.predict(X)
        err = 1 - np.sum(preds == y) / preds.shape[0]

        if err == 0:  # Avoid division by zero
            return np.inf
        if err == 1:  # Completely wrong model
            return -np.inf

        amount_of_say = np.log((1 - err) / err) + np.log(K - 1)
        return amount_of_say

    def _fit_base_learner(self, X_bootstrapped, y_bootstrapped):
        """Trains a weak learner and calculates its weight."""
        base_learner = self.weak_learner(max_depth=1)
        base_learner.fit(X_bootstrapped, y_bootstrapped)
        base_learner.amount_of_say = self._calculate_amount_of_say(base_learner, self.X_train, self.y_train)
        return base_learner

    def _update_dataset(self, sample_weights):
        """Creates bootstrapped samples w.r.t. sample weights."""
        n_samples = self.X_train.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        X_bootstrapped = self.X_train[bootstrap_indices]
        y_bootstrapped = self.y_train[bootstrap_indices]
        return X_bootstrapped, y_bootstrapped

    def _calculate_sample_weights(self, base_learner):
        """Calculates sample weights for boosting."""
        preds = base_learner.predict(self.X_train)
        incorrect = (preds != self.y_train).astype(int)
        sample_weights = np.exp(base_learner.amount_of_say * incorrect)
        sample_weights /= np.sum(sample_weights)  # Normalize
        return sample_weights

    def fit(self, X, y):
        """
        Train the AdaBoost ensemble.

        Parameters:
        - X: Training features, shape (n_samples, n_features)
        - y: Training labels, shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.label_count = len(np.unique(y))
        sample_weights = np.full(X.shape[0], 1 / X.shape[0])

        self.learners = []
        self.alphas = []

        for _ in range(self.num_learners):
            X_bootstrapped, y_bootstrapped = self._update_dataset(sample_weights)
            base_learner = self._fit_base_learner(X_bootstrapped, y_bootstrapped)

            if base_learner.amount_of_say == np.inf:  # Perfect classifier
                self.learners.append(base_learner)
                self.alphas.append(1)
                break

            self.learners.append(base_learner)
            self.alphas.append(base_learner.amount_of_say)

            sample_weights = self._calculate_sample_weights(base_learner)

    def _predict_scores_w_base_learners(self, X):
        """Aggregates predictions from all base learners."""
        pred_scores = np.zeros((self.num_learners, X.shape[0], self.label_count))
        for idx, base_learner in enumerate(self.learners):
            pred_probs = base_learner.predict_probabilities(X)
            pred_scores[idx] = pred_probs * self.alphas[idx]
        return pred_scores

    def predict_proba(self, X):
        """Returns predicted probabilities for input data."""
        base_learners_pred_scores = self._predict_scores_w_base_learners(X)
        avg_base_learners_pred_scores = np.mean(base_learners_pred_scores, axis=0)
        column_sums = np.sum(avg_base_learners_pred_scores, axis=1)
        pred_probs = avg_base_learners_pred_scores / column_sums[:, np.newaxis]
        return pred_probs

    def predict(self, X):
        """Predicts labels for input data."""
        pred_probs = self.predict_proba(X)
        return np.argmax(pred_probs, axis=1)

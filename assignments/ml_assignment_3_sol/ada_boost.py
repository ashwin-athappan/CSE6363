import numpy as np
from decision_tree import DecisionTree

class AdaBoost():

    def __init__(self, weak_learner=DecisionTree, num_learners=10, learning_rate=1.0) -> None:
        """
        Initialize the AdaBoost classifier.

        :param weak_learner: Base weak learner class (default: DecisionTree with max_depth=1).
        :param num_learners: Number of base learners in the ensemble.
        :param learning_rate: Controls the weight update step.
        """
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.learners = []
        self.alphas = []

    def _calculate_amount_of_say(self, error: float) -> float:
        """
        Calculates the weight (alpha) for a weak learner based on its error.

        :param error: The classification error of the weak learner.
        :return: The computed alpha value.
        """
        K = self.label_count
        return self.learning_rate * (np.log((1 - error) / error) + np.log(K - 1))

    def _update_weights(self, sample_weights: np.array, alpha: float, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Updates sample weights based on the classifier's performance.

        :param sample_weights: Current sample weights.
        :param alpha: Weight of the weak learner.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Updated sample weights.
        """
        incorrect = (y_pred != y_true).astype(int)
        sample_weights *= np.exp(alpha * incorrect)
        return sample_weights / np.sum(sample_weights)

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """
        Train the AdaBoost model.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.label_count = len(np.unique(y_train))
        n_samples = X_train.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.num_learners):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            x_bootstrap, y_bootstrap = X_train.to_numpy()[bootstrap_indices], y_train.to_numpy()[bootstrap_indices]

            # Train weak learner
            learner = self.weak_learner()
            learner.fit(x_bootstrap, y_bootstrap)

            # Calculate error and alpha
            y_pred = learner.predict(X_train)
            error = np.sum(sample_weights * (y_pred != y_train)) / np.sum(sample_weights)
            if error >= 1 - 1/self.label_count or error == 0:
                continue  # Skip this iteration if error is too high or zero

            alpha = self._calculate_amount_of_say(error)
            sample_weights = self._update_weights(sample_weights, alpha, y_train, y_pred)

            # Store the weak learner and its alpha value
            self.learners.append(learner)
            self.alphas.append(alpha)

    def predict_probabilities(self, X: np.array) -> np.array:
        """
        Predict class probabilities for given data.

        :param X: Input features.
        :return: Class probability distribution.
        """
        pred_scores = np.zeros((X.shape[0], self.label_count))

        for learner, alpha in zip(self.learners, self.alphas):
            pred_probs = learner.predict_probabilities(X)
            pred_scores += alpha * pred_probs

        pred_probs = pred_scores / np.sum(pred_scores, axis=1, keepdims=True)
        return pred_probs

    def predict(self, X: np.array) -> np.array:
        """
        Predict class labels for given data.

        :param X: Input features.
        :return: Predicted class labels.
        """
        pred_probs = self.predict_probabilities(X)
        return np.argmax(pred_probs, axis=1)

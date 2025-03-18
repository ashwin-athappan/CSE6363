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

    def fit(self, X, y):
        """
        Train the AdaBoost ensemble.

        Parameters:
        - X: Training features, shape (n_samples, n_features)
        - y: Training labels, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        # Ensure labels are in {-1, +1}
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("AdaBoost requires binary classification with labels in {-1, +1}.")
        y = np.where(y == classes[0], -1, 1)

        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.num_learners):
            # Train a weak learner on weighted samples
            tree = self.weak_learner(max_depth=1)  # Stump (depth=1)
            tree.fit(X, y, sample_weight=sample_weights)

            # Get predictions and error
            preds = tree.predict(X)
            err = np.sum(sample_weights[preds != y])

            if err == 0:  # Perfect classifier
                self.learners = [tree]
                self.alphas = [1]
                return

            # Compute learner weight (alpha)
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / (err + 1e-10))

            # Update sample weights
            sample_weights *= np.exp(-alpha * y * preds)
            sample_weights /= np.sum(sample_weights)  # Normalize

            # Store weak learner and its weight
            self.learners.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predicts class labels for input samples.

        Parameters:
        - X: Input features, shape (n_samples, n_features)

        Returns:
        - Predicted labels, shape (n_samples,)
        """
        X = np.array(X)
        # Weighted sum of weak learner predictions
        pred_sum = sum(alpha * learner.predict(X) for alpha, learner in zip(self.alphas, self.learners))

        # Apply sign function to get final prediction
        return np.sign(pred_sum)

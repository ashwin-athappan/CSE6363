from typing import Any

import numpy as np
from decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    """
    Random Forest Classifier
    Training: Use "fit" method with training set features and labels
    Predicting: Use "predict" method with test set features
    """

    def __init__(self, n_trees=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0,
                 min_features=2, bootstrap_sample_size=None) -> None:
        """
        Initialize RandomForest parameters.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.min_features = min_features
        self.bootstrap_sample_size = bootstrap_sample_size
        self.trees = []
        self.feature_subsets = []

    def _create_bootstrap_samples(self, X, y) -> tuple[list[Any], list[Any]]:
        """
        Creates bootstrap samples for each tree.
        """
        bootstrap_X, bootstrap_y = [], []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X.append(X.iloc[sample_indices].values)  # Ensure it's converted to NumPy array
            bootstrap_y.append(y.iloc[sample_indices].values)  # Convert labels to NumPy array

        return bootstrap_X, bootstrap_y

    def _select_random_features(self, num_features):
        """
        Selects a random subset of features for a tree.
        """
        selected_features = np.random.choice(num_features, size=np.random.randint(self.min_features, num_features + 1), replace=False)
        return selected_features

    def fit(self, X_train, y_train):
        """
        Trains the RandomForest by fitting each DecisionTree on a bootstrap sample and random subset of features.
        """
        num_features = X_train.shape[1]
        bootstrap_X, bootstrap_y = self._create_bootstrap_samples(X_train, y_train)

        for i in range(self.n_trees):
            selected_features = self._select_random_features(num_features)
            self.feature_subsets.append(selected_features)

            X_subset = bootstrap_X[i][:, selected_features]
            y_subset = bootstrap_y[i]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                min_information_gain=self.min_information_gain)
            tree.fit(X_subset, y_subset)

            self.trees.append(tree)

    def predict(self, X_test):
        """
        Predicts class labels using majority voting across all trees.
        """
        tree_predictions = np.array([
            tree.predict(X_test.iloc[:, self.feature_subsets[i]]) for i, tree in enumerate(self.trees)
        ])

        # Perform majority voting
        final_predictions = [Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X_test.shape[0])]

        return np.array(final_predictions)

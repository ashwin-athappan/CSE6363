import numpy as np
from collections import Counter
from tree_node import TreeNode

from graphviz import Digraph

class DecisionTree:
    """
    Decision Tree Classifier with multiple splitting criteria.
    """

    def __init__(self, max_depth=4, min_samples_leaf=1,
                 min_samples_split=0.0, num_features_splitting=None,
                 criterion='entropy', amount_of_say=None, feature_names=None,
                 min_information_gain=0.0) -> None:
        """
        Parameters:
        max_depth: int -> max depth of the tree
        min_samples_leaf: int -> minimum samples per leaf to allow splitting
        min_samples_split: float -> minimum samples to allow splitting
        num_features_splitting: str -> feature selection strategy ("sqrt", "log", or None)
        criterion: str -> splitting criterion ('gini', 'entropy', or 'misclassification')
        amount_of_say: float -> used for Adaboost algorithm
        feature_names: list -> List of feature names for interpretability in visualization.
        min_information_gain: float -> minimum information gain to allow splitting
        """
        self.feature_importance = None
        self.tree = None
        self.labels_in_train = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.num_features_splitting = num_features_splitting
        self.criterion = criterion
        self.amount_of_say = amount_of_say
        self.feature_names = feature_names
        self.min_information_gain = min_information_gain

    def _gini(self, class_probabilities: list) -> float:
        return 1 - sum([p ** 2 for p in class_probabilities])

    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    def _misclassification(self, class_probabilities: list) -> float:
        return 1 - max(class_probabilities)

    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        return self._entropy(self._class_probabilities(labels))

    def _data_gini(self, labels: list) -> float:
        return self._gini(self._class_probabilities(labels))

    def _data_misclassification(self, labels: list) -> float:
        return self._misclassification(self._class_probabilities(labels))

    def _partition(self, subsets: list) -> float:
        total_count = sum([len(subset) for subset in subsets])
        if self.criterion == 'entropy':
            return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
        elif self.criterion == 'gini':
            return sum([self._data_gini(subset) * (len(subset) / total_count) for subset in subsets])
        elif self.criterion == 'misclassification':
            return sum([self._data_misclassification(subset) * (len(subset) / total_count) for subset in subsets])
        else:
            raise ValueError("Unsupported criterion: must be 'entropy', 'gini', or 'misclassification'.")

    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2

    def _select_features_to_use(self, data: np.array) -> list:
        feature_idx = list(range(data.shape[1] - 1))
        if self.num_features_splitting == "sqrt":
            return np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.num_features_splitting == "log":
            return np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            return feature_idx

    def _find_best_split(self, data: np.array) -> tuple:
        g1_min, g2_min = None, None
        min_entropy_feature_idx, min_entropy_feature_val = None, None
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data)

        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                g1, g2 = self._split(data, idx, feature_val)
                part_entropy = self._partition([g1[:, -1], g2[:, -1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode | None:
        if current_depth > self.max_depth:
            return None

        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        label_probabilities = self._find_label_probs(data)
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy

        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Stopping conditions
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        elif information_gain < self.min_information_gain:  # Add min_information_gain check
            return node
        elif information_gain < self.min_samples_split:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        return node

    def _predict_one_sample(self, X: np.array) -> np.array:
        pred_probs = None
        node = self.tree

        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def fit(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        self.tree = self._create_tree(data=train_data, current_depth=0)
        self.feature_importance = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        self.feature_importance = {k: v / total for total in (sum(self.feature_importance.values()),) for k, v in
                                   self.feature_importance.items()}

    def predict_probabilities(self, X_set: np.array) -> np.array:
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        pred_probs = self.predict_probabilities(X_set)
        if pred_probs.shape[1] == 1:  # Binary classification
            return (pred_probs[:, 0] > 0.5).astype(int)
        else:
            return np.argmax(pred_probs, axis=1)

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node is not None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)
    def visualize_tree(self):
        """
        Visualizes the decision tree using Graphviz and displays it inside Jupyter Notebook.
        """
        if self.tree is None:
            print("The tree has not been trained yet.")
            return

        dot = Digraph()
        dot.attr(size="50,10")
        self._add_nodes_edges(self.tree, dot)
        return dot  # This will render in Jupyter Notebook automatically

    def _add_nodes_edges(self, node, dot, parent_name=None):
        """
        Recursively adds nodes and edges to the Graphviz object, including the splitting class.
        """
        if node is None:
            return

        # Ensure feature_names is properly indexed
        feature_name = (
            self.feature_names[node.feature_idx]
            if self.feature_names is not None and len(self.feature_names) > node.feature_idx
            else f"Feature {node.feature_idx}"
        )

        node_label = (
            f"{feature_name}\n≤ {node.feature_val:.3f}\n"
            f"InfoGain: {node.information_gain:.3f}\n"
            f"Splitting Class: {"Survived" if node.majority_class == 0.0 else "Died"}"
        )
        node_name = str(id(node))  # Unique identifier for Graphviz

        dot.node(node_name, node_label, shape='box')

        if parent_name:
            dot.edge(parent_name, node_name)

        # Recursively add left and right children
        self._add_nodes_edges(node.left, dot, node_name)
        self._add_nodes_edges(node.right, dot, node_name)



    def _calculate_feature_importance(self, node):
        if node is not None:
            self.feature_importance[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)

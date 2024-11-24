import numpy as np
import torch

class DecisionTree:
    def __init__(self, max_depth=50, min_samples_split=2, min_samples_leaf=1, num_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_thresholds = num_thresholds
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or len(set(y)) == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return leaf_value

        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return self._most_common_label(y)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return self._most_common_label(y)

        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, num_features):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(num_features):
            thresholds = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), self.num_thresholds)
            for threshold in thresholds:
                gini = self._gini_index(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_index(self, feature_column, y, threshold):
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float('inf')
        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])
        weighted_gini = (len(left_idxs) * left_gini + len(right_idxs) * right_gini) / len(y)
        return weighted_gini

    def _gini(self, y):
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _split(self, feature_column, threshold):
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            if inputs[feature] <= threshold:
                node = left
            else:
                node = right
        return node
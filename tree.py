import numpy as np

# Decision Tree class implementation
class DecisionTree:
    # Initialize the DecisionTree object with the specified hyperparameters
    def __init__(self, max_depth=50, min_samples_split=2, min_samples_leaf=1, num_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_thresholds = num_thresholds
        self.tree = None

    # Method to train the decision tree on the given data (x: features, y: labels)
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    # Recursive function to build the decision tree from the given data
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # Check stopping criteria max depth, all labels are the same, or not enough samples
        if depth >= self.max_depth or len(set(y)) == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        # Find the best split for the current node
        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return self._most_common_label(y)
        
        # Split data into left and right branches based on the best split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        # Check if leaf nodes have enough samples
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return self._most_common_label(y)

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return the best split 
        return (best_feature, best_threshold, left_subtree, right_subtree)

    # Find the best split for the current node
    def _best_split(self, X, y, num_features):
        # Initialize variables
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        # Iterate over all features and thresholds
        for feature in range(num_features):
            thresholds = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), self.num_thresholds)
            # Evaluate gini index for each threshold
            for threshold in thresholds:
                gini = self._gini_index(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        # return best split
        return best_feature, best_threshold

    # Calculate the gini index for a given feature and threshold
    def _gini_index(self, feature_column, y, threshold):
        # Split data into left and right subtrees
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]

        # Check if split results in empty left or right subtrees
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float('inf')
        
        # Calculate weighted gini index for left and right subtrees
        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])

        # Average weighted gini index
        weighted_gini = (len(left_idxs) * left_gini + len(right_idxs) * right_gini) / len(y)
        return weighted_gini

    # Calculate the gini index
    def _gini(self, y):
     
        y = y.numpy()
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    # Split data into left and right subtrees
    def _split(self, feature_column, threshold):
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        return left_idxs, right_idxs

    # Find the most common label in the current node
    def _most_common_label(self, y):

        y = y.numpy()
        return np.bincount(y).argmax()

    # Predict labels for a dataset
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    # Predict label for a single input
    def _predict(self, inputs):
        # Traverse the decision tree to find the leaf node
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            if inputs[feature] <= threshold:
                node = left
            else:
                node = right
        return node
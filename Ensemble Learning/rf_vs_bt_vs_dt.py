import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, criterion='information_gain', max_features=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_types = self._determine_feature_types(X)
        self.root = self._grow_tree(X, y)

    def _determine_feature_types(self, X):
        feature_types = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
                feature_types.append("numerical")
            else:
                feature_types.append("categorical")
        return feature_types

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_labels = len(y), len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1:
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(np.arange(self.n_features), size=self.max_features, replace=False)
        feat_idxs = np.atleast_1d(feat_idxs)

        best_feat, best_value = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X, best_feat, best_value)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_value, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_criteria = -np.inf
        split_idx, split_value = None, None

        for feat in feat_idxs:
            if self.feature_types[feat] == "numerical":
                values = X[:, feat]
                median = np.median(values)
                left_idxs = np.where(X[:, feat] <= median)[0]
                right_idxs = np.where(X[:, feat] > median)[0]
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                criteria_value = self._calc_criteria(y, left_idxs, right_idxs)
                if criteria_value > best_criteria:
                    best_criteria = criteria_value
                    split_idx = feat
                    split_value = median
            else:
                unique_values = np.unique(X[:, feat])
                for val in unique_values:
                    left_idxs, right_idxs = self._split(X, feat, val)
                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                        continue
                    criteria_value = self._calc_criteria(y, left_idxs, right_idxs)
                    if criteria_value > best_criteria:
                        best_criteria = criteria_value
                        split_idx = feat
                        split_value = val

        if split_idx is None:
            return None, None

        return split_idx, split_value

    def _calc_criteria(self, y, left_idxs, right_idxs):
        left_y, right_y = y[left_idxs], y[right_idxs]
        if len(left_y) == 0 or len(right_y) == 0:
            return 0

        if self.criterion == 'information_gain':
            return self._information_gain(y, left_y, right_y)

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        parent_value = self._calculate_criteria_value(y)
        left_value = self._calculate_criteria_value(left_y)
        right_value = self._calculate_criteria_value(right_y)
        child_value = (len(left_y) / n) * left_value + (len(right_y) / n) * right_value
        return parent_value - child_value

    def _calculate_criteria_value(self, y):
        return self._entropy(y)

    def _entropy(self, y):
        label_counts = Counter(y)
        total = len(y)
        entropy_value = 0.0

        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy_value -= p * np.log2(p)

        return entropy_value

    def _split(self, X, feat, value):
        if self.feature_types[feat] == "numerical":
            left_idxs = np.where(X[:, feat] <= value)[0]
            right_idxs = np.where(X[:, feat] > value)[0]
        else:
            left_idxs = np.where(X[:, feat] == value)[0]
            right_idxs = np.where(X[:, feat] != value)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if self.feature_types[node.feature] == "numerical":
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=500, criterion='information_gain', max_features=None):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            bootstrapped_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[bootstrapped_indices]
            y_sample = y[bootstrapped_indices]

            tree = DecisionTree(max_depth=np.inf, criterion=self.criterion, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        majority_vote = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
        return majority_vote

class BaggedTrees:
    def __init__(self, n_trees=500, criterion='information_gain'):
        self.n_trees = n_trees
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            bootstrapped_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[bootstrapped_indices]
            y_sample = y[bootstrapped_indices]

            tree = DecisionTree(max_depth=np.inf, criterion=self.criterion)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        majority_vote = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
        return majority_vote

class BiasVarianceDecomposition:
    def __init__(self, n_repeats=100, n_trees=500, sample_size=1000, is_random_forest=True, max_features=None):
        self.n_repeats = n_repeats
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.is_random_forest = is_random_forest 
        self.max_features = max_features

    def fit(self, X_train, y_train, X_test, y_test):
        n_test_samples = X_test.shape[0]
        single_tree_preds = np.zeros((self.n_repeats, n_test_samples))
        forest_preds = np.zeros((self.n_repeats, n_test_samples))

        for i in range(self.n_repeats):
            sampled_indices = np.random.choice(len(X_train), self.sample_size, replace=False)
            X_sample, y_sample = X_train[sampled_indices], y_train[sampled_indices]
            
            if self.is_random_forest:
                forest = RandomForest(n_trees=self.n_trees, max_features=self.max_features)
            else:
                forest = BaggedTrees(n_trees=self.n_trees)
            
            forest.fit(X_sample, y_sample)

            single_tree = forest.trees[0]
            
            single_tree_preds[i, :] = single_tree.predict(X_test)
            forest_preds[i, :] = forest.predict(X_test)

        single_tree_avg_preds = np.mean(single_tree_preds, axis=0)
        single_tree_bias = np.mean((single_tree_avg_preds - y_test) ** 2)
        single_tree_variance = np.mean(np.var(single_tree_preds, axis=0))
        single_tree_error = single_tree_bias + single_tree_variance

        forest_avg_preds = np.mean(forest_preds, axis=0)
        forest_bias = np.mean((forest_avg_preds - y_test) ** 2)
        forest_variance = np.mean(np.var(forest_preds, axis=0))
        forest_error = forest_bias + forest_variance

        return {
            'single_tree': {'bias': single_tree_bias, 'variance': single_tree_variance, 'error': single_tree_error},
            'forest': {'bias': forest_bias, 'variance': forest_variance, 'error': forest_error}
        }

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("datasets/bank/test.csv", names=column_headers)

y_train = df_train['label'].apply(lambda x: 1 if x == 'yes' else -1).values
y_test = df_test['label'].apply(lambda x: 1 if x == 'yes' else -1).values

X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

rf_bvd = BiasVarianceDecomposition(n_repeats=100, n_trees=500, sample_size=1000, is_random_forest=True, max_features=4)
rf_results = rf_bvd.fit(X_train, y_train, X_test, y_test)

bt_bvd = BiasVarianceDecomposition(n_repeats=100, n_trees=500, sample_size=1000, is_random_forest=False)
bt_results = bt_bvd.fit(X_train, y_train, X_test, y_test)

print(f"Single Random Tree - Bias: {rf_results['single_tree']['bias']:.4f}, Variance: {rf_results['single_tree']['variance']:.4f}, Squared Error: {rf_results['single_tree']['error']:.4f}")
print(f"Random Forest - Bias: {rf_results['forest']['bias']:.4f}, Variance: {rf_results['forest']['variance']:.4f}, Squared Error: {rf_results['forest']['error']:.4f}")

print(f"Single Bagged Tree - Bias: {bt_results['single_tree']['bias']:.4f}, Variance: {bt_results['single_tree']['variance']:.4f}, Squared Error: {bt_results['single_tree']['error']:.4f}")
print(f"Bagged Trees - Bias: {bt_results['forest']['bias']:.4f}, Variance: {bt_results['forest']['variance']:.4f}, Squared Error: {bt_results['forest']['error']:.4f}")

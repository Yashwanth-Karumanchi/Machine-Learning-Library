import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    def __init__(self, max_depth=100, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_types = self._determine_feature_types(X)
        self.root = self._grow_tree(X, y)

    # Function to determine if the attribute is of type categorical or numerical
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

        feat_idxs = np.arange(self.n_features)
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
        elif self.criterion == 'gini':
            return self._gini(y, left_y, right_y)
        elif self.criterion == 'majority_error':
            return self._majority_error(y, left_y, right_y)

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        parent_value = self._calculate_criteria_value(y)
        left_value = self._calculate_criteria_value(left_y)
        right_value = self._calculate_criteria_value(right_y)
        child_value = (len(left_y) / n) * left_value + (len(right_y) / n) * right_value
        return parent_value - child_value

    def _calculate_criteria_value(self, y):
        if self.criterion == 'information_gain':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini_index(y)
        elif self.criterion == 'majority_error':
            return self._majority_error_value(y)

    def _gini(self, y, left_y, right_y):
        parent_gini = self._gini_index(y)
        gini_left = self._gini_index(left_y)
        gini_right = self._gini_index(right_y)
        n = len(y)
        child_gini = (len(left_y) / n) * gini_left + (len(right_y) / n) * gini_right
        return parent_gini - child_gini

    def _majority_error(self, y, left_y, right_y):
        parent_error = self._majority_error_value(y)
        majority_error_left = self._majority_error_value(left_y)
        majority_error_right = self._majority_error_value(right_y)
        n = len(y)
        child_error = (len(left_y) / n) * majority_error_left + (len(right_y) / n) * majority_error_right
        return parent_error - child_error

    def _split(self, X, feat, value):
        if self.feature_types[feat] == "numerical":
            left_idxs = np.where(X[:, feat] <= value)[0]
            right_idxs = np.where(X[:, feat] > value)[0]
        else:
            left_idxs = np.where(X[:, feat] == value)[0]
            right_idxs = np.where(X[:, feat] != value)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        label_counts = Counter(y)
        total = len(y)
        entropy_value = 0.0

        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy_value -= p * np.log2(p)

        return entropy_value

    def _gini_index(self, y):
        counts = Counter(y)
        total = len(y)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    def _majority_error_value(self, y):
        counts = Counter(y)
        total = len(y)
        return 1 - max(counts.values()) / total

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

# BaggedTrees class
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

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("datasets/bank/test.csv", names=column_headers)

y_train = df_train['label'].apply(lambda x: 1 if x == 'yes' else -1).values
y_test = df_test['label'].apply(lambda x: 1 if x == 'yes' else -1).values

X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

n_trees_max = 500
train_errors = []
test_errors = []

for n_trees in range(1, n_trees_max + 1):
    bagged_trees = BaggedTrees(n_trees=n_trees, criterion='information_gain')
    bagged_trees.fit(X_train, y_train)

    y_train_pred = bagged_trees.predict(X_train)
    y_test_pred = bagged_trees.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)
    if n_trees % 50 == 0:
        print(f"Trees: {n_trees} train Error: {train_error}------test Error: {test_error}")


print(f"Final Training Error after {n_trees_max} trees: {train_errors[-1]:.4f}")
print(f"Final Test Error after {n_trees_max} trees: {test_errors[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_trees_max + 1), train_errors, label='Training Error', linestyle='-')
plt.plot(range(1, n_trees_max + 1), test_errors, label='Test Error', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Bagged Trees: Training and Test Errors vs. Number of Trees')
plt.legend()
plt.tight_layout()
plt.show()

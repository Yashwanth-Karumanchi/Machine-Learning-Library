import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=100, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y, sample_weights=None):
        self.n_features = X.shape[1]
        self.feature_types = self._determine_feature_types(X)
        self.root = self._grow_tree(X, y, sample_weights)

    def _determine_feature_types(self, X):
        feature_types = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
                feature_types.append("numerical")
            else:
                feature_types.append("categorical")
        return feature_types

    def _grow_tree(self, X, y, sample_weights, depth=0):
        n_samples, n_labels = len(y), len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1:
            return Node(value=self._most_common_label(y, sample_weights))

        feat_idxs = np.arange(self.n_features)
        best_feat, best_value = self._best_split(X, y, feat_idxs, sample_weights)

        if best_feat is None:
            return Node(value=self._most_common_label(y, sample_weights))

        left_idxs, right_idxs = self._split(X, best_feat, best_value)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y, sample_weights))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], sample_weights[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], sample_weights[right_idxs], depth + 1)

        return Node(best_feat, best_value, left, right)

    def _most_common_label(self, y, sample_weights):
        weighted_counts = Counter()
        for label, weight in zip(y, sample_weights):
            weighted_counts[label] += weight
        return weighted_counts.most_common(1)[0][0]

    def _best_split(self, X, y, feat_idxs, sample_weights):
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
                criteria_value = self._calc_criteria(y, left_idxs, right_idxs, sample_weights)
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
                    criteria_value = self._calc_criteria(y, left_idxs, right_idxs, sample_weights)
                    if criteria_value > best_criteria:
                        best_criteria = criteria_value
                        split_idx = feat
                        split_value = val

        return split_idx, split_value

    def _calc_criteria(self, y, left_idxs, right_idxs, sample_weights):
        left_y, right_y = y[left_idxs], y[right_idxs]
        left_weights = sample_weights[left_idxs]
        right_weights = sample_weights[right_idxs]

        if self.criterion == 'information_gain':
            return self._information_gain(y, left_y, right_y, sample_weights, left_weights, right_weights)

    def _information_gain(self, y, left_y, right_y, parent_weights, left_weights, right_weights):
        parent_entropy = self._entropy(y, parent_weights)
        left_entropy = self._entropy(left_y, left_weights)
        right_entropy = self._entropy(right_y, right_weights)
        weighted_avg_child_entropy = (
            np.sum(left_weights) * left_entropy + np.sum(right_weights) * right_entropy
        ) / np.sum(parent_weights)
        return parent_entropy - weighted_avg_child_entropy

    def _entropy(self, y, sample_weights):
        total_weight = np.sum(sample_weights)
        if total_weight == 0:
            return 0
        counts = Counter(y)
        entropy_value = 0.0
        for label in counts:
            p = (sample_weights[y == label].sum()) / total_weight
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

# DecisionStump class: a specialized DecisionTree with max_depth=1
class DecisionStump(DecisionTree):
    def __init__(self, criterion='information_gain'):
        super().__init__(max_depth=1, criterion=criterion)

# AdaBoost class
class AdaBoost:
    def __init__(self, n_clf=500):
        self.n_clf = n_clf
        self.clfs = []
        self.clf_weights = []
        self.training_errors = []
        self.test_errors = []
        self.stump_errors = []
        self.formatted_stump_errors = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        sample_weights = np.ones(n_samples) / n_samples

        for t in range(self.n_clf):
            clf = DecisionStump()
            clf.fit(X, y, sample_weights)
            y_pred = clf.predict(X)

            error = np.sum(sample_weights[y != y_pred]) / np.sum(sample_weights)

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.clfs.append(clf)
            self.clf_weights.append(alpha)

            sample_weights *= np.exp(-alpha * y * y_pred)
            sample_weights /= np.sum(sample_weights)

            train_predictions = self.predict(X_train, n_clf=t+1)
            test_predictions = self.predict(X_test, n_clf=t+1)
            train_error = 1 - accuracy_score(y_train, train_predictions)
            test_error = 1 - accuracy_score(y_test, test_predictions)
            self.training_errors.append(train_error)
            self.test_errors.append(test_error)

            self.stump_errors.append(error / n_samples)
        self.formatted_stump_errors = [f"{err:.6f}" for err in self.stump_errors]

    def predict(self, X, n_clf=None):
        if n_clf is None:
            n_clf = len(self.clfs)
        clf_preds = np.array([clf.predict(X) for clf in self.clfs[:n_clf]])
        return np.sign(np.dot(self.clf_weights[:n_clf], clf_preds))

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("datasets/bank/test.csv", names=column_headers)

y_train = df_train['label'].apply(lambda x: 1 if x == 'yes' else -1).values
y_test = df_test['label'].apply(lambda x: 1 if x == 'yes' else -1).values

X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

n_clf = 500
ada = AdaBoost(n_clf=n_clf)
ada.fit(X_train, y_train)

print(f"Final AdaBoost training error: {ada.training_errors[-1]:.4f}")
print(f"Final AdaBoost test error: {ada.test_errors[-1]:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_clf + 1), ada.training_errors, label="Training Error", linestyle='-')
plt.plot(range(1, n_clf + 1), ada.test_errors, label="Test Error", linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Training and Test Errors vs. Iteration (T)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_clf + 1), ada.formatted_stump_errors, label="Decision Stump Error", linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Decision Stump Errors vs. Iteration")
plt.legend()

plt.tight_layout()
plt.show()
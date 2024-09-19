import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate

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
        self.majority_values = self._impute_missing_values(X, y)
        X = self._replace_missing_values(X)
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

    def _impute_missing_values(self, X, y):
        majority_values = {}
        for i in range(X.shape[1]):
            if self.feature_types[i] == "categorical":
                values = X[:, i]
                # Replace 'unknown' with the majority value
                if 'unknown' in values:
                    majority_value = Counter(values[values != 'unknown']).most_common(1)[0][0]
                    majority_values[i] = majority_value
        return majority_values

    def _replace_missing_values(self, X):
        X_imputed = X.copy()
        for i in range(X.shape[1]):
            if i in self.majority_values:
                X_imputed[X[:, i] == 'unknown', i] = self.majority_values[i]
        return X_imputed

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
                median = np.median(values.astype(float))
                left_idxs = np.where(values <= median)[0]
                right_idxs = np.where(values > median)[0]
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
                    if val == 'unknown':
                        continue
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
            left_idxs = np.where(X[:, feat].astype(float) <= value)[0]
            right_idxs = np.where(X[:, feat].astype(float) > value)[0]
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
        X = self._replace_missing_values(X)
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

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("datasets/bank/test.csv", names=column_headers)

# Impute missing values in training set
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

metrics = {crit: {'train': [], 'test': []} for crit in ['information_gain', 'majority_error', 'gini']}
best_depths = {crit: 0 for crit in metrics.keys()}
best_accs = {crit: 0 for crit in metrics.keys()}

max_depth = int(input("Enter maximum tree depth: "))

for depth in range(1, max_depth + 1):
    for criterion in metrics.keys():
        clf = DecisionTree(max_depth=depth, criterion=criterion)
        clf.fit(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        train_acc = np.mean(y_train == y_train_pred)
        test_acc = np.mean(y_test == y_test_pred)
        
        metrics[criterion]['train'].append(round(1 - train_acc, 3))
        metrics[criterion]['test'].append(round(1 - test_acc, 3))
        
        if test_acc > best_accs[criterion]:
            best_accs[criterion] = test_acc
            best_depths[criterion] = depth

table_data = []
for depth in range(1, len(metrics['information_gain']['train']) + 1):
    row = [depth,
           metrics['information_gain']['train'][depth - 1],
           metrics['information_gain']['test'][depth - 1],
           metrics['majority_error']['train'][depth - 1],
           metrics['majority_error']['test'][depth - 1],
           metrics['gini']['train'][depth - 1],
           metrics['gini']['test'][depth - 1]]

    if depth == best_depths['information_gain']:
        row = ['*' + str(depth),
               '*' + str(metrics['information_gain']['train'][depth - 1]),
               '*' + str(metrics['information_gain']['test'][depth - 1]),
               str(metrics['majority_error']['train'][depth - 1]),
               str(metrics['majority_error']['test'][depth - 1]),
               str(metrics['gini']['train'][depth - 1]),
               str(metrics['gini']['test'][depth - 1])]
    elif depth == best_depths['majority_error']:
        row = [str(depth),
               str(metrics['information_gain']['train'][depth - 1]),
               str(metrics['information_gain']['test'][depth - 1]),
               '*' + str(metrics['majority_error']['train'][depth - 1]),
               '*' + str(metrics['majority_error']['test'][depth - 1]),
               str(metrics['gini']['train'][depth - 1]),
               str(metrics['gini']['test'][depth - 1])]
    elif depth == best_depths['gini']:
        row = [str(depth),
               str(metrics['information_gain']['train'][depth - 1]),
               str(metrics['information_gain']['test'][depth - 1]),
               str(metrics['majority_error']['train'][depth - 1]),
               str(metrics['majority_error']['test'][depth - 1]),
               '*' + str(metrics['gini']['train'][depth - 1]),
               '*' + str(metrics['gini']['test'][depth - 1])]
        
    table_data.append(row)

headers = ["Depth", "I.G(Train)", "I.G(Test)", "M.E(Train)", "M.E(Test)", "Gini(Train)", "Gini(Test)"]
print(tabulate(table_data, headers=headers, tablefmt="pretty"))

for crit in ['information_gain', 'majority_error', 'gini']:
    print(f"Least {crit.replace('_', ' ').title()} error observed at depth {best_depths[crit]} with test loss {1-best_accs[crit]:.3f}.")
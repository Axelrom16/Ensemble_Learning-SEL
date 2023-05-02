import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import random


#
# Random Forest
#
class RandomForestClassifier:
    def __init__(self, dataset_name, target_name, cat_variables, test_split, n_estimators=100, number_features=None,
                 max_depth=100, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.encoding = None
        self.p = None
        self.n = None
        self.X_test = None
        self.class_names = None
        self.num_classes = None
        self.attr_class = None
        self.attr_names = None
        self.X = None
        self.out_file = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.name = dataset_name
        self.cat_variables = cat_variables
        self.target_name = target_name
        self.test_split = test_split
        self.nf = number_features

    def import_data(self):
        """
        Import the data set using the dataset_name parameter, process the data and perform the train/test split
        :return:
        """
        data = pd.read_csv('./data/' + self.name + '.csv', na_values=['?'])

        # Missing data
        if data.isnull().values.any():
            col_missing = data.columns[np.where(data.isnull().sum() > 0)]
            for col in col_missing:
                if col not in self.cat_variables:
                    data[col].fillna((data[col].mean()), inplace=True)
                else:
                    data = data.fillna(data.mode().iloc[0])

        # Encode categorical variables
        le = LabelEncoder()
        encoding = []
        for cat in self.cat_variables:
            data[cat] = le.fit_transform(data[cat])
            encoding.append(dict(zip(le.classes_, range(len(le.classes_)))))
        self.encoding = encoding

        # Values type
        for cat in data.columns:
            data[cat] = data[cat].astype(object)

        # Train/Test split
        y = data[self.target_name]
        X = data.drop(self.target_name, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split, random_state=42,
                                                            shuffle=True)
        X_train[self.target_name] = y_train
        X_train = X_train.reset_index(drop=True)
        X_test[self.target_name] = y_test
        X_test = X_test.reset_index(drop=True)

        self.X = X_train
        self.X = self.X[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.attr_names = self.X.columns
        self.attr_class = list(self.X.dtypes)
        self.num_classes = len(np.unique(self.X[self.target_name]))
        self.class_names = np.unique(self.X[self.target_name])
        self.X_test = X_test
        self.X_test = self.X_test[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        if self.nf is None:
            self.nf = self.X.shape[1] - 1

        # Write info data
        results_file = open("results/" + self.name + "/rf/train_" + self.name + ".txt", "w")
        self.out_file = results_file
        self.out_file.write("#-------\n")
        self.out_file.write("Data\n")
        self.out_file.write("#-------\n")
        self.out_file.write("Train data shape: " + str(self.X.shape) + "\n")
        self.out_file.write("Test data shape: " + str(self.X_test.shape) + "\n")
        self.out_file.write("Encoding: \n")
        i = 0
        for enc in encoding:
            self.out_file.write("\t" + self.cat_variables[i] + ': ' + str(enc) + "\n")
            i += 1

        return self.X.drop(self.target_name, axis=1), self.X[self.target_name],\
               self.X_test.drop(self.target_name, axis=1), self.X_test[self.target_name]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for i in tqdm(range(self.n_estimators), total=self.n_estimators):
            # Bootstrap sample of data
            indices_row = np.random.choice(X.shape[0], X.shape[0], replace=True)
            indices_col = np.random.choice(X.shape[1], self.nf, replace=False)
            X_bootstrap = X.iloc[indices_row, indices_col]
            y_bootstrap = y.iloc[indices_row]
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(attr_names=self.attr_names, cat_variables=self.cat_variables,
                                          max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return stats.mode(predictions, axis=0)[0][0]

    def evaluate(self, X, y):
        # Accuracy train
        pred_train = self.predict(self.X)
        acc_train = np.sum(pred_train == self.X[self.target_name]) / self.X.shape[0]
        # Accuracy test
        pred = self.predict(X)
        acc = np.sum(pred == y) / X.shape[0]
        pred = [int(x) for x in pred]
        # Confusion matrix
        cm = confusion_matrix(list(y), pred)
        target_names = list(self.encoding[-1].keys())
        target_names = [str(x) for x in target_names]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names,
                    ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("results/" + self.name + "/rf/cm_" + self.name + "_nt" + str(self.n_estimators) + "_nf" + str(self.nf) + ".svg")
        plt.clf()
        # Classification report
        with open("results/" + self.name + "/rf/report_" + self.name + "_nt" + str(self.n_estimators) + "_nf" + str(self.nf) + ".txt", 'w') as f:
            f.write(classification_report(list(y), pred, target_names=target_names))

        return acc_train, acc

    def show_trees(self):
        trees_dict = [x.tree for x in self.trees]
        return trees_dict

    def count_features(self, tree):
        feature_counts = {}
        stack = [tree]

        while stack:
            node = stack.pop()
            feature = node['feature']

            if feature not in feature_counts:
                feature_counts[feature] = 0

            feature_counts[feature] += 1

            if isinstance(node['left'], dict):
                stack.append(node['left'])

            if isinstance(node['right'], dict):
                stack.append(node['right'])

        return feature_counts

    def feature_importance(self):
        trees_dict = self.show_trees()

        count_vec = []
        for tree in trees_dict:
            count_vec.append(self.count_features(tree))

        counter = Counter(count_vec[0])
        for c in range(len(count_vec)-1):
            counter.update(Counter(count_vec[c+1]))
        final_counter = dict(counter)

        return final_counter


#
# Decision Forest
#
class DecisionForestClassifier:
    def __init__(self, dataset_name, target_name, cat_variables, test_split, n_estimators=100, number_features=None,
                 max_depth=80, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.encoding = None
        self.p = None
        self.n = None
        self.X_test = None
        self.class_names = None
        self.num_classes = None
        self.attr_class = None
        self.attr_names = None
        self.X = None
        self.out_file = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.name = dataset_name
        self.cat_variables = cat_variables
        self.target_name = target_name
        self.test_split = test_split
        self.nf = number_features

    def import_data(self):
        """
        Import the data set using the dataset_name parameter, process the data and perform the train/test split
        :return:
        """
        data = pd.read_csv('./data/' + self.name + '.csv', na_values=['?'])

        # Missing data
        if data.isnull().values.any():
            col_missing = data.columns[np.where(data.isnull().sum() > 0)]
            for col in col_missing:
                if col not in self.cat_variables:
                    data[col].fillna((data[col].mean()), inplace=True)
                else:
                    data = data.fillna(data.mode().iloc[0])

        # Encode categorical variables
        le = LabelEncoder()
        encoding = []
        for cat in self.cat_variables:
            data[cat] = le.fit_transform(data[cat])
            encoding.append(dict(zip(le.classes_, range(len(le.classes_)))))
        self.encoding = encoding

        # Values type
        for cat in data.columns:
            data[cat] = data[cat].astype(object)

        # Train/Test split
        y = data[self.target_name]
        X = data.drop(self.target_name, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split, random_state=42,
                                                            shuffle=True)
        X_train[self.target_name] = y_train
        X_train = X_train.reset_index(drop=True)
        X_test[self.target_name] = y_test
        X_test = X_test.reset_index(drop=True)

        self.X = X_train
        self.X = self.X[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.attr_names = self.X.columns
        self.attr_class = list(self.X.dtypes)
        self.num_classes = len(np.unique(self.X[self.target_name]))
        self.class_names = np.unique(self.X[self.target_name])
        self.X_test = X_test
        self.X_test = self.X_test[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        if self.nf is None:
            self.nf = self.X.shape[1] - 1

        # Write info data
        results_file = open("results/" + self.name + "/df/train_" + self.name + ".txt", "w")
        self.out_file = results_file
        self.out_file.write("#-------\n")
        self.out_file.write("Data\n")
        self.out_file.write("#-------\n")
        self.out_file.write("Train data shape: " + str(self.X.shape) + "\n")
        self.out_file.write("Test data shape: " + str(self.X_test.shape) + "\n")
        self.out_file.write("Encoding: \n")
        i = 0
        for enc in encoding:
            self.out_file.write("\t" + self.cat_variables[i] + ': ' + str(enc) + "\n")
            i += 1

        return self.X.drop(self.target_name, axis=1), self.X[self.target_name],\
               self.X_test.drop(self.target_name, axis=1), self.X_test[self.target_name]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        runif = False
        if self.nf == 'runif':
            runif = True
        for i in tqdm(range(self.n_estimators), total=self.n_estimators):
            if runif:
                temp_nf = int(round(np.random.uniform(1, X.shape[1])))
            else:
                temp_nf = int(self.nf)
            indices_col = np.random.choice(X.shape[1], temp_nf, replace=False)
            X_new = X.iloc[:, indices_col]
            y_new = y
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(attr_names=self.attr_names, cat_variables=self.cat_variables,
                                          max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_new, y_new)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return stats.mode(predictions, axis=0)[0][0]

    def evaluate(self, X, y):
        # Accuracy train
        pred_train = self.predict(self.X)
        acc_train = np.sum(pred_train == self.X[self.target_name]) / self.X.shape[0]
        # Accuracy test
        pred = self.predict(X)
        acc = np.sum(pred == y) / X.shape[0]
        pred = [int(x) for x in pred]
        # Confusion matrix
        cm = confusion_matrix(list(y), pred)
        target_names = list(self.encoding[-1].keys())
        target_names = [str(x) for x in target_names]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names,
                    ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("results/" + self.name + "/df/cm_" + self.name + "_nt" + str(self.n_estimators) + "_nf" + str(self.nf) + ".svg")
        plt.clf()
        # Classification report
        with open("results/" + self.name + "/df/report_" + self.name + "_nt" + str(self.n_estimators) + "_nf" + str(self.nf) + ".txt", 'w') as f:
            f.write(classification_report(list(y), pred, target_names=target_names))

        return acc_train, acc

    def show_trees(self):
        trees_dict = [x.tree for x in self.trees]
        return trees_dict

    def count_features(self, tree):
        feature_counts = {}
        stack = [tree]

        while stack:
            node = stack.pop()
            feature = node['feature']

            if feature not in feature_counts:
                feature_counts[feature] = 0

            feature_counts[feature] += 1

            if isinstance(node['left'], dict):
                stack.append(node['left'])

            if isinstance(node['right'], dict):
                stack.append(node['right'])

        return feature_counts

    def feature_importance(self):
        trees_dict = self.show_trees()

        count_vec = []
        for tree in trees_dict:
            count_vec.append(self.count_features(tree))

        counter = Counter(count_vec[0])
        for c in range(len(count_vec)-1):
            counter.update(Counter(count_vec[c+1]))
        final_counter = dict(counter)

        return final_counter


#
# Decision Tree
#
class DecisionTreeClassifier:
    def __init__(self, attr_names, cat_variables, max_depth=80, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.attr_names = attr_names
        self.cat_variables = cat_variables

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.tree) for _, x in X.iterrows()])

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        # Base cases: if only one class label remains, or if maximum depth or minimum samples per split is reached
        if len(np.unique(y)) == 1 or depth == self.max_depth or num_samples < self.min_samples_split:
            return np.argmax(np.bincount(y))
        # Find best split by iterating over features and values
        best_feature = 0
        best_value = 0
        best_score = -1
        for feature in range(num_features):
            if X.columns[feature] in self.cat_variables:
                for value in np.unique(X.iloc[:, feature]):
                    left_indices = np.where(X.iloc[:, feature] == value)[0]
                    right_indices = np.where(X.iloc[:, feature] != value)[0]
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue
                    score = self.gini_index(y.iloc[left_indices], y.iloc[right_indices], num_labels)
                    if score > best_score:
                        best_feature = feature
                        best_value = value
                        best_score = score
            else:
                sort_values = sorted(list(X.iloc[:, feature]), reverse=True)
                sort_values = np.unique(sort_values)
                if len(sort_values) > 100:
                    sort_values = sort_values[:len(sort_values):20]
                values = []
                for i in range(len(sort_values)-1):
                    values.append(np.mean(sort_values[i:(i+2)]))
                for value in values:
                    left_indices = np.where(X.iloc[:, feature] <= value)[0]
                    right_indices = np.where(X.iloc[:, feature] > value)[0]
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue
                    score = self.gini_index(y.iloc[left_indices], y.iloc[right_indices], num_labels)
                    if score > best_score:
                        best_feature = feature
                        best_value = value
                        best_score = score
        if best_score == -1:
            return np.argmax(np.bincount(y))
        # Create subtree for left and right nodes
        if X.columns[best_feature] in self.cat_variables:
            left_indices = np.where(X.iloc[:, best_feature] == best_value)[0]
            right_indices = np.where(X.iloc[:, best_feature] != best_value)[0]
        else:
            left_indices = np.where(X.iloc[:, best_feature] <= best_value)[0]
            right_indices = np.where(X.iloc[:, best_feature] > best_value)[0]
        left_subtree = self.build_tree(X.iloc[left_indices], y.iloc[left_indices], depth+1)
        right_subtree = self.build_tree(X.iloc[right_indices], y.iloc[right_indices], depth+1)
        return {'feature': X.columns[best_feature], 'value': best_value, 'left': left_subtree, 'right': right_subtree}

    def gini_index(self, y_left, y_right, num_labels):
        num_left = len(y_left)
        num_right = len(y_right)
        score_left = 1.0 - sum([(np.sum(y_left == label) / num_left) ** 2 for label in range(num_labels)])
        score_right = 1.0 - sum([(np.sum(y_right == label) / num_right) ** 2 for label in range(num_labels)])
        return (num_left * score_left + num_right * score_right) / (num_left + num_right)

    def traverse_tree(self, x, tree):
        if isinstance(tree, (int, np.integer)):
            return tree
        if x[tree['feature']] <= tree['value']:
            return self.traverse_tree(x, tree['left'])
        else:
            return self.traverse_tree(x, tree['right'])

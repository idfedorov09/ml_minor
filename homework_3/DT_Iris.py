import os
import sys

import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def show_plot(y_true, y_pred, filename=None, accuracy=None):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if accuracy is not None:
        plt.text(0.5, 1.1, f'Accuracy: {accuracy:.2f}', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, color='red', fontweight='bold')

    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True)
    if filename:
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def entropy(class_probabilities):
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0)

def data_entropy(labels):
    return entropy(class_probabilities(labels))

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

class DataSet:
    def __init__(self, X=np.array([[]]), Y=np.array([])):
        self.X_data = X
        self.Y_data = Y
        self.feature_names = None

class Node:
    def __init__(self, indx = -1, value = None):
        self.feature_indx = indx
        self.feature_name = ""
        self.node_value = value
        self.right_child = None
        self.left_child = None

class DTree:
    def __init__(self):
        self.root = Node()
        self.train_data = None
        self.class_data = None
        self.predicted = np.array([])
        self.data_names = np.array([])

    def split_by(self, f_indx, fval,  DSet):
        npoints, nfeatures = DSet.X_data.shape

        DSet_left = DataSet()
        DSet_left.feature_names = DSet.feature_names.delete(f_indx)
        DSet_right = DataSet()
        DSet_right.feature_names = DSet.feature_names.delete(f_indx)

        for fentry, target_class in zip(DSet.X_data, DSet.Y_data):
            if type(fentry[f_indx]) != str:
                with_f_deleted = np.delete(fentry, f_indx)
                if fentry[f_indx] < fval:
                    DSet_left.Y_data = np.append(DSet_left.Y_data, target_class)
                    DSet_left.X_data = np.append(DSet_left.X_data, with_f_deleted)
                else:
                    DSet_right.X_data = np.append(DSet_right.X_data, with_f_deleted)
                    DSet_right.Y_data = np.append(DSet_right.Y_data, target_class)

        DSet_right.X_data = DSet_right.X_data.reshape(len(DSet_right.Y_data), nfeatures-1)
        DSet_left.X_data = DSet_left.X_data.reshape(len(DSet_left.Y_data), nfeatures - 1)

        return DSet_left, DSet_right

    def decide(self, DSet, decision_node):
        npoints, nfeatures = DSet.X_data.shape

        if np.any(DSet.X_data) == False: # DSet.X_data is empty
            label_counts = Counter(DSet.Y_data)
            most_common_label = label_counts.most_common(1)[0][0]
            decision_node.feature_indx = -1
            decision_node.node_value = most_common_label
            return

        if len(np.unique(DSet.Y_data)) == 1: # if only one class in dataset
            decision_node.feature_indx = -1
            decision_node.node_value = DSet.Y_data[0]
            return

        min_entropy = sys.maxsize  # min entropy of splitting

        DSetLeft = DataSet()
        DSetRight = DataSet()

        for findx in range(0, nfeatures):  # итеррируем по всем признакам
            x = DSet.X_data[:, findx]  # select current feature column
            x_unique_values = np.unique(x)  # remove duplicates from current column of features

            for current_feature_value in x_unique_values:  # смотрим разбиение по каждому уникальному признаку
                data_set_left, data_set_right = self.split_by(findx, current_feature_value, DSet) # разбиваем по каждому уникальному признаку
                entropy_of_partition = partition_entropy([data_set_left.Y_data, data_set_right.Y_data]) # считаем энтропию разбиения
                if min_entropy > entropy_of_partition:
                    min_entropy = entropy_of_partition
                    decision_node.node_value = current_feature_value
                    decision_node.feature_name = DSet.feature_names[findx]
                    decision_node.feature_indx = np.where(self.data_names == decision_node.feature_name)[0][0]
                    DSetLeft = data_set_left
                    DSetRight = data_set_right

        decision_node.left_child = Node()
        decision_node.right_child = Node()
        self.decide(DSetLeft, decision_node.left_child)
        self.decide(DSetRight, decision_node.right_child)

    def traverse(self, entry, node, columns):
        if node.left_child is None and node.right_child is None:
            self.predicted = np.append(self.predicted, node.node_value)
            return

        if type(node.node_value) != str:
            indx = columns.index(node.feature_name)
            if entry[indx] < node.node_value:
                self.traverse(entry, node.left_child, columns)
            else:
                self.traverse(entry, node.right_child, columns)

    def classify(self, data, answers=None, mark=""):
        columns = data.columns.to_list()
        X_data = data.to_numpy()

        self.predicted = np.array([])
        for entry in X_data:
            if type(self.root.node_value) != str:
                self.traverse(entry, self.root, columns)

        if type(answers) is not None:
            answers_data = answers.to_numpy()
            le = LabelEncoder()
            answers_data = le.fit_transform(answers_data)
            accuracy = 0
            for p, r in zip(self.predicted, answers_data):
                if p == r:
                    accuracy+=1

            print(f"{mark}accuracy : ", accuracy/len(self.predicted))
            return accuracy/len(self.predicted)

    def build_tree(self, train_data, target_data):
        self.root = Node()
        self.data_names = train_data.columns.to_numpy()
        self.train_data = train_data
        self.class_data = target_data

        X_data = train_data.to_numpy()
        Y_data = target_data.to_numpy()

        le = LabelEncoder()
        Y_data = le.fit_transform(Y_data)

        MainDataSet = DataSet(X_data, Y_data)
        MainDataSet.feature_names = train_data.columns

        self.decide(MainDataSet, self.root)

class ForestEvaluator:
    def __init__(self, X, y, trees_sizes):
        self.current_tree = -1
        self.X, self.y = X, y
        self.trees_sizes = trees_sizes

    def _cur_folder_name(self):
        return f'forest_{self.current_tree}/'

    def _path_to_save(self, filename):
        return self._cur_folder_name() + filename

    def _random_forest_evaluate(self, trees_count):
        self.current_tree += 1
        trees_results = []
        _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

        for i in range(trees_count):
            # Bagging: выборка с возвращением для обучения каждого дерева
            X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.33, random_state=i)

            # Создаем и обучаем дерево решений
            Tree = DTree()
            Tree.build_tree(X_train, y_train)
            cur_accuracy = Tree.classify(X_test, y_test)
            trees_results.append(Tree.predicted)
            show_plot(y_test, Tree.predicted, filename=self._path_to_save(f'tree#{i}'), accuracy=cur_accuracy)

        predictions_df = pd.DataFrame(trees_results).T
        mode_prediction = predictions_df.mode(axis=1)[0]

        accuracy = accuracy_score(y_test, mode_prediction)
        print(f'RF#{self.current_tree} TOTAL accuracy:' + str(accuracy))
        show_plot(y_test, mode_prediction, filename=self._path_to_save(f'RF#{self.current_tree}'), accuracy=accuracy)


    def run(self):
        for i in self.trees_sizes:
            self._random_forest_evaluate(i)

def main():
    target_feature = "Star type"

    data = pd.read_csv("Stars.csv")
    X = data.drop(["Star color", "Spectral Class", "Star type"], axis=1)
    y = data[target_feature]
    forest_evaluator = ForestEvaluator(X, y, range(1, 3))
    forest_evaluator.run()


if __name__ == '__main__':
    sys.stdout = open('result.txt', 'w')
    main()
    sys.stdout.close()
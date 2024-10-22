import sys
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    def split_by(self, f_indx, fval, DSet):
        DSet_left = DataSet()
        DSet_left.feature_names = DSet.feature_names.drop(DSet.feature_names[f_indx])
        DSet_right = DataSet()
        DSet_right.feature_names = DSet.feature_names.drop(DSet.feature_names[f_indx])

        for fentry, class_type in zip(DSet.X_data, DSet.Y_data):
            with_f_deleted = np.delete(fentry, f_indx)
            if fentry[f_indx] < fval:
                DSet_left.Y_data = np.append(DSet_left.Y_data, class_type)
                if DSet_left.X_data.size == 0:
                    DSet_left.X_data = np.array([with_f_deleted])
                else:
                    DSet_left.X_data = np.vstack([DSet_left.X_data, with_f_deleted])
            else:
                DSet_right.Y_data = np.append(DSet_right.Y_data, class_type)
                if DSet_right.X_data.size == 0:
                    DSet_right.X_data = np.array([with_f_deleted])
                else:
                    DSet_right.X_data = np.vstack([DSet_right.X_data, with_f_deleted])

        return DSet_left, DSet_right

    def decide(self, DSet, decision_node):
        if DSet.X_data.size == 0:
            label_counts = Counter(DSet.Y_data)
            most_common_label = label_counts.most_common(1)[0][0]
            decision_node.feature_indx = -1
            decision_node.node_value = most_common_label
            return

        if len(np.unique(DSet.Y_data)) == 1:
            decision_node.feature_indx = -1
            decision_node.node_value = DSet.Y_data[0]
            return

        npoints, nfeatures = DSet.X_data.shape
        min_entropy = sys.maxsize

        DSetLeft = DataSet()
        DSetRight = DataSet()

        for findx in range(nfeatures):
            x = DSet.X_data[:, findx]
            x_unique_values = np.unique(x)

            for current_feature_value in x_unique_values:
                data_set_left, data_set_right = self.split_by(findx, current_feature_value, DSet)
                entropy_of_partition = partition_entropy([data_set_left.Y_data, data_set_right.Y_data])
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

        indx = columns.index(node.feature_name)
        if entry[indx] < node.node_value:
            self.traverse(entry, node.left_child, columns)
        else:
            self.traverse(entry, node.right_child, columns)

    def classify(self, data, answers=None):
        columns = data.columns.to_list()
        X_data = data.to_numpy()

        self.predicted = np.array([])
        for entry in X_data:
            self.traverse(entry, self.root, columns)

        if answers is not None:
            answers_data = answers.to_numpy()
            le = LabelEncoder()
            answers_data = le.fit_transform(answers_data)
            eff = 0
            for p, r in zip(self.predicted, answers_data):
                print(" predicted : ", p, " real : ", r)
                if p == r:
                    eff+=1

            print(" efficiency : ", eff/len(self.predicted))
            cf_matrix = confusion_matrix(answers_data, self.predicted)
            sns.heatmap(cf_matrix, annot=True)
            plt.show()

    def build_tree(self, train_data, class_data):
        self.root = Node()
        self.data_names = train_data.columns.to_numpy()
        self.train_data = train_data
        self.class_data = class_data

        X_data = train_data.to_numpy()
        Y_data = class_data.to_numpy()

        MainDataSet = DataSet(X_data, Y_data)
        MainDataSet.feature_names = train_data.columns

        self.decide(MainDataSet, self.root)

def main():
    data = pd.read_csv("Stars.csv")

    # Исключаем строковые столбцы
    X = data.drop(["Star color", "Spectral Class", "Star type"], axis=1)
    y = data["Star type"]

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Создаем и обучаем дерево решений
    Tree = DTree()
    Tree.build_tree(X_train, y_train)

    # Классифицируем тестовую выборку и выводим метрики
    Tree.classify(X_test, y_test)

if __name__ == '__main__':
    sys.stdout = open('result.txt', 'w')
    main()
    sys.stdout.close()
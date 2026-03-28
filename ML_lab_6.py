import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["name"])
    X = df.drop("status", axis=1)
    y = df["status"]
    return X, y

def entropy(y):
    total = len(y)
    counts = Counter(y)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * log2(p)
    return ent


# Equal Width Binning
def equal_width_binning(data, bins=4):
    data = np.array(data)
    min_val, max_val = np.min(data), np.max(data)
    width = (max_val - min_val) / bins
    binned = np.floor((data - min_val) / width)
    binned[binned == bins] = bins - 1
    return binned.astype(int)


# Equal Frequency Binning
def equal_frequency_binning(data, bins=4):
    data = np.array(data)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(data, quantiles)
    return np.digitize(data, edges[1:-1])


# General Binning (A4)
def binning(data, bins=4, method="equal_width"):
    if method == "equal_width":
        return equal_width_binning(data, bins)
    elif method == "equal_frequency":
        return equal_frequency_binning(data, bins)
    else:
        raise ValueError("Invalid method")


# Apply binning to dataset
def bin_dataset(X, bins=4, method="equal_width"):
    X_binned = X.copy()
    for col in X.columns:
        X_binned[col] = binning(X[col], bins, method)
    return X_binned


#  A2: Gini Index 
def gini_index(y):
    total = len(y)
    counts = Counter(y)
    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    return gini


# A3: Information Gain 
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = np.unique(X[feature])
    weighted_entropy = 0

    for val in values:
        subset_y = y[X[feature] == val]
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * entropy(subset_y)

    return total_entropy - weighted_entropy


def find_best_feature(X, y):
    gains = {}
    for col in X.columns:
        gains[col] = information_gain(X, y, col)
    best_feature = max(gains, key=gains.get)
    return best_feature, gains


#  A5: Custom Decision Tree 
class Node:
    def __init__(self, feature=None, results=None, children=None):
        self.feature = feature
        self.results = results
        self.children = children


def build_tree(X, y):
    if len(set(y)) == 1:
        return Node(results=y.iloc[0])

    if X.empty:
        return Node(results=Counter(y).most_common(1)[0][0])

    best_feature, _ = find_best_feature(X, y)
    tree = Node(feature=best_feature, children={})

    for val in np.unique(X[best_feature]):
        subset_X = X[X[best_feature] == val].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == val]
        tree.children[val] = build_tree(subset_X, subset_y)

    return tree


# A6: Visualization
def visualize_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)

    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.title("Decision Tree")
    plt.show()

    return model


#  A7: Decision Boundary 
def decision_boundary(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
    plt.title("Decision Boundary")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()

    return model


# MAIN FUNCTION
def main():
    X, y = load_data("parkinsons.csv")

    # Binning
    X_binned = bin_dataset(X, bins=4, method="equal_width")

    # A1: Entropy
    ent = entropy(y)
    print("Entropy:", ent)

    # A2: Gini Index
    gini = gini_index(y)
    print("Gini Index:", gini)

    # A3: Best Feature
    best_feature, gains = find_best_feature(X_binned, y)
    print("Best Feature (Root Node):", best_feature)
    print("Information Gain of all features:", gains)

    # A5: Build Tree
    tree = build_tree(X_binned, y)
    print("Decision Tree Built Successfully")

    # A6: Visualize Tree
    visualize_tree(X_binned, y)

    # A7: Decision Boundary (2 features)
    decision_boundary(X_binned.iloc[:, :2], y)


#  RUN 
if __name__ == "__main__":
    main()
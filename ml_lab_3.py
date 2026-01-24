import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def manual_dot(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def euclidean_norm(A):
    total = 0
    for x in A:
        total += x * x
    return math.sqrt(total)

def euclidean_distance(A, B):
    total = 0
    for i in range(len(A)):
        total += (A[i] - B[i]) ** 2
    return math.sqrt(total)


def manual_mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

def manual_variance(values):
    mu = manual_mean(values)
    total = 0
    for v in values:
        total += (v - mu) ** 2
    return total / len(values)

def manual_std(values):
    return math.sqrt(manual_variance(values))

def mean_vector(matrix):
    mean_vec = []
    for j in range(len(matrix[0])):
        column = []
        for i in range(len(matrix)):
            column.append(matrix[i][j])
        mean_vec.append(manual_mean(column))
    return mean_vec


def minkowski_distance(A, B, p):
    total = 0
    for i in range(len(A)):
        total += abs(A[i] - B[i]) ** p
    return total ** (1 / p)


def train_test_split_manual(X, y, test_ratio):
    split_index = int(len(X) * (1 - test_ratio))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def knn_train(X_train, y_train, k):
    return {"X": X_train, "y": y_train, "k": k}

def knn_predict(model, test_point):
    distances = []
    for i in range(len(model["X"])):
        d = euclidean_distance(model["X"][i], test_point)
        distances.append((d, model["y"][i]))

    distances.sort(key=lambda x: x[0])

    votes = {}
    for i in range(model["k"]):
        label = distances[i][1]
        votes[label] = votes.get(label, 0) + 1

    best_label = None
    max_votes = -1
    for label in votes:
        if votes[label] > max_votes:
            max_votes = votes[label]
            best_label = label

    return best_label

def knn_predict_all(model, X_test):
    predictions = []
    for x in X_test:
        predictions.append(knn_predict(model, x))
    return predictions

def knn_accuracy_manual(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def manual_confusion_matrix(y_true, y_pred):
    TP = FP = TN = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN

def precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

def main():
    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [6, 5],
        [7, 8],
        [8, 7]
    ]
    y = [0, 0, 0, 1, 1, 1]

    A = [1, 2, 3]
    B = [4, 5, 6]
    print("Manual Dot:", manual_dot(A, B))
    print("Manual Norm A:", euclidean_norm(A))
    print("Manual Norm B:", euclidean_norm(B))
    print("NumPy Dot:", np.dot(A, B))
    print("NumPy Norm A:", np.linalg.norm(A))

    class0 = X[:3]
    class1 = X[3:]
    mean0 = mean_vector(class0)
    mean1 = mean_vector(class1)
    print("Class 0 Mean:", mean0)
    print("Class 1 Mean:", mean1)
    print("Inter-class Distance:", euclidean_distance(mean0, mean1))

    feature = [row[0] for row in X]
    print("Feature Mean:", manual_mean(feature))
    print("Feature Variance:", manual_variance(feature))
    plt.hist(feature, bins=5)
    plt.title("Feature Histogram")
    plt.show()

    distances = []
    for p in range(1, 11):
        distances.append(minkowski_distance(X[0], X[1], p))
    print("Minkowski Distances:", distances)
    plt.plot(range(1, 11), distances)
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, 0.33)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print()

    sklearn_model = KNeighborsClassifier(n_neighbors=3)
    sklearn_model.fit(X_train, y_train)
    print("Sklearn Accuracy:", sklearn_model.score(X_test, y_test))
    sklearn_preds = sklearn_model.predict(X_test)
    print("Sklearn Predictions:", sklearn_preds)
    print()

    manual_model = knn_train(X_train, y_train, 3)
    manual_preds = knn_predict_all(manual_model, X_test)
    manual_acc = knn_accuracy_manual(y_test, manual_preds)
    print("Manual kNN Accuracy:", manual_acc)
    print()

    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        model = knn_train(X_train, y_train, k)
        preds = knn_predict_all(model, X_test)
        accuracies.append(knn_accuracy_manual(y_test, preds))

    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.show()

    TP, FP, TN, FN = manual_confusion_matrix(y_test, manual_preds)
    p = precision(TP, FP)
    r = recall(TP, FN)
    f1 = f1_score(p, r)

    print("Confusion Matrix:")
    print("TP:", TP, "FP:", FP)
    print("FN:", FN, "TN:", TN)
    print("Precision:", p)
    print("Recall:", r)
    print("F1 Score:", f1)

main()

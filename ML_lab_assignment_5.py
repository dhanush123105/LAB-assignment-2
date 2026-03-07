import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


def load_dataset(path):
    df = pd.read_csv(path)
    return df



def prepare_data(df, target):
    X = df.drop(columns=["name", target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# A1 Linear regression using one feature
def single_feature_regression(X_train, y_train, feature):

    X = X_train[[feature]]

    model = LinearRegression()
    model.fit(X, y_train)

    y_pred = model.predict(X)

    return model, y_pred


# Metrics calculation
def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2


# A3 Linear regression using all features
def multiple_feature_regression(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    return model, y_pred


# A4 KMeans clustering
def perform_kmeans(X, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)

    return kmeans.labels_, kmeans.cluster_centers_


# A5 Clustering scores
def clustering_metrics(X, labels):

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return sil, ch, db


# A6 Evaluate multiple k values
def evaluate_k(X, k_range):

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_range:

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return sil_scores, ch_scores, db_scores



def plot_scores(k_range, sil, ch, db):

    plt.plot(k_range, sil)
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()

    plt.plot(k_range, ch)
    plt.title("Calinski Harabasz Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()

    plt.plot(k_range, db)
    plt.title("Davies Bouldin Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()


# A7 Elbow method
def elbow_method(X, k_range):

    distortions = []

    for k in k_range:

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)

        distortions.append(kmeans.inertia_)

    return distortions


def plot_elbow(k_range, distortions):

    plt.plot(k_range, distortions)
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.show()



# MAIN FUNCTION
def main():

    df = load_dataset("parkinsons.csv")

    target = "PPE"

    X_train, X_test, y_train, y_test = prepare_data(df, target)

    # A1
    feature = X_train.columns[0]
    model_single, train_pred = single_feature_regression(X_train, y_train, feature)

    test_pred = model_single.predict(X_test[[feature]])

    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    print("A1 & A2 Single Feature Regression")
    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)


    # A3
    model_multi, train_pred_multi = multiple_feature_regression(X_train, y_train)

    test_pred_multi = model_multi.predict(X_test)

    train_metrics_multi = calculate_metrics(y_train, train_pred_multi)
    test_metrics_multi = calculate_metrics(y_test, test_pred_multi)

    print("\nA3 Multiple Feature Regression")
    print("Train Metrics:", train_metrics_multi)
    print("Test Metrics:", test_metrics_multi)


    # A4
    X_cluster = df.drop(columns=["name", target])

    labels, centers = perform_kmeans(X_cluster, 2)

    print("\nCluster Centers")
    print(centers)


    # A5
    sil, ch, db = clustering_metrics(X_cluster, labels)

    print("\nClustering Scores")
    print("Silhouette:", sil)
    print("CH Score:", ch)
    print("DB Index:", db)


    # A6
    k_range = range(2, 10)

    sil_scores, ch_scores, db_scores = evaluate_k(X_cluster, k_range)

    plot_scores(k_range, sil_scores, ch_scores, db_scores)


    # A7
    distortions = elbow_method(X_cluster, range(2, 20))

    plot_elbow(range(2, 20), distortions)


# Run main
if __name__ == "__main__":
    main()
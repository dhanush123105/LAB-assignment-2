import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Optional (if installed)
try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None


#    Load Dataset 
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["name"])
    X = df.drop("status", axis=1)
    y = df["status"]
    return X, y


#Preprocessing-
def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


#  Split 
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


#  Evaluation 
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred)
    }

    return results


#  A2: Hyperparameter Tuning 
def tune_model(model, param_dist, X_train, y_train):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


#A3: Multiple Models 
def run_all_models(X_train, X_test, y_train, y_test):
    results = {}

    # SVM
    svm = SVC()
    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
    svm_best = tune_model(svm, svm_params, X_train, y_train)
    results["SVM"] = evaluate_model(svm_best, X_train, X_test, y_train, y_test)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt_params = {
        "max_depth": [3, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    dt_best = tune_model(dt, dt_params, X_train, y_train)
    results["Decision Tree"] = evaluate_model(dt_best, X_train, X_test, y_train, y_test)

    # Random Forest
    rf = RandomForestClassifier()
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }
    rf_best = tune_model(rf, rf_params, X_train, y_train)
    results["Random Forest"] = evaluate_model(rf_best, X_train, X_test, y_train, y_test)

    # AdaBoost
    ada = AdaBoostClassifier()
    ada_params = {
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1]
    }
    ada_best = tune_model(ada, ada_params, X_train, y_train)
    results["AdaBoost"] = evaluate_model(ada_best, X_train, X_test, y_train, y_test)

    # Naive Bayes
    nb = GaussianNB()
    results["Naive Bayes"] = evaluate_model(nb, X_train, X_test, y_train, y_test)

    # MLP
    mlp = MLPClassifier(max_iter=500)
    mlp_params = {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"]
    }
    mlp_best = tune_model(mlp, mlp_params, X_train, y_train)
    results["MLP"] = evaluate_model(mlp_best, X_train, X_test, y_train, y_test)

    # XGBoost (if available)
    if XGBClassifier:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_params = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5]
        }
        xgb_best = tune_model(xgb, xgb_params, X_train, y_train)
        results["XGBoost"] = evaluate_model(xgb_best, X_train, X_test, y_train, y_test)

    return results


#  MAIN 
def main():
    X, y = load_data("parkinsons.csv")

    X = preprocess(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    results = run_all_models(X_train, X_test, y_train, y_test)

    # Print Results (ONLY HERE)
    print("\nModel Performance Comparison:\n")

    for model, res in results.items():
        print(f"{model}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
        print()


if __name__ == "__main__":
    main()
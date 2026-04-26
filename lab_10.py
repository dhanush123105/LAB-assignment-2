# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# LIME & SHAP
from lime.lime_tabular import LimeTabularExplainer
import shap

# LOAD DATA

def load_data(file_path):
    data = pd.read_csv('parkinsons.csv')

    X = data.drop(columns=['status', 'name'])
    y = data['status']

    return X, y



# A1: CORRELATION HEATMAP

def correlation_analysis(X):
    """
    Generate correlation matrix and heatmap
    """
    corr_matrix = X.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")

    return corr_matrix



# A2: PCA (99% variance)

def pca_99_model(X_train, X_test, y_train, y_test):
    """
    PCA with 99% variance
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    return acc, X_train_pca.shape[1]



# A3: PCA (95% variance)

def pca_95_model(X_train, X_test, y_train, y_test):
    """
    PCA with 95% variance
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    return acc, X_train_pca.shape[1]



# A4: SEQUENTIAL FEATURE SELECTION

def sequential_feature_selection(X_train, X_test, y_train, y_test):
    """
    Select best features using Sequential Selection
    """

    model = RandomForestClassifier()

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=10,
        direction='forward'
    )

    sfs.fit(X_train, y_train)

    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    model.fit(X_train_sfs, y_train)

    y_pred = model.predict(X_test_sfs)
    acc = accuracy_score(y_test, y_pred)

    return acc, X_train_sfs.shape[1]



# A5: LIME EXPLANATION

def lime_explanation(model, X_train, X_test):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Healthy', 'Parkinson'],
        mode='classification'
    )

    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba
    )

    return exp



# A5: SHAP EXPLANATION

def shap_explanation(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    return shap_values


# MAIN FUNCTION

def main():
    file_path = "parkinsons (1).csv"

    X, y = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # A1
   
    correlation_analysis(X)
    print("A1: Correlation heatmap displayed")

 
    # A2

    acc_99, comp_99 = pca_99_model(X_train, X_test, y_train, y_test)
    print("A2: PCA 99% -> Accuracy:", acc_99, "Features:", comp_99)

    # A3
  
    acc_95, comp_95 = pca_95_model(X_train, X_test, y_train, y_test)
    print("A3: PCA 95% -> Accuracy:", acc_95, "Features:", comp_95)

    # A4
    
    acc_sfs, feat_sfs = sequential_feature_selection(
        X_train, X_test, y_train, y_test
    )
    print("A4: SFS -> Accuracy:", acc_sfs, "Features:", feat_sfs)

   
    # Base model for explanation

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # A5: LIME
   
    lime_exp = lime_explanation(model, X_train, X_test)
    print("A5: LIME Explanation:", lime_exp.as_list())

    
    # A5: SHAP
 
    shap_values = shap_explanation(model, X_train)
    print("A5: SHAP values computed")

    # SHAP plot (outside function)
    shap.summary_plot(shap_values, X_train)



# RUN

if __name__ == "__main__":
    main()
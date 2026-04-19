
# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# LIME
from lime.lime_tabular import LimeTabularExplainer



# DATA LOADING FUNCTION

def load_data(file_path):
    """
    Load dataset and split into features and target
    """
    data = pd.read_csv("parkinsons.csv")

    # Parkinson dataset: target column is 'status'
    X = data.drop(columns=['status', 'name'])
    y = data['status']

    return X, y


# A1: STACKING CLASSIFIER

def stacking_model(X_train, X_test, y_train, y_test):
    """
    Implements stacking classifier with different base models
    """

    # Base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ]

    # Meta model
    meta_model = LogisticRegression()

    # Stacking classifier
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    # Train
    stack_model.fit(X_train, y_train)

    # Predict
    y_pred = stack_model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    return stack_model, acc



# A2: PIPELINE IMPLEMENTATION

def pipeline_model(X_train, X_test, y_train, y_test):
    """
    Pipeline with scaling + classifier
    """

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    return pipeline, acc



# A3: LIME EXPLAINER

def lime_explanation(pipeline, X_train, X_test):
    """
    Explain predictions using LIME
    """

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Healthy', 'Parkinson'],
        mode='classification'
    )

    # Explain first test instance
    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        pipeline.predict_proba
    )

    return exp


# MAIN FUNCTION

def main():
    # Load dataset
    file_path = "parkinsons (1).csv"
    X, y = load_data(file_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # A1: Stacking
   
    stack_model, stack_acc = stacking_model(X_train, X_test, y_train, y_test)
    print("A1: Stacking Accuracy:", stack_acc)

  
    # A2: Pipeline
   
    pipe_model, pipe_acc = pipeline_model(X_train, X_test, y_train, y_test)
    print("A2: Pipeline Accuracy:", pipe_acc)

    
    # A3: LIME
   
    explanation = lime_explanation(pipe_model, X_train, X_test)
    print("A3: LIME Explanation:")
    print(explanation.as_list())


# RUN PROGRAM

if __name__ == "__main__":
    main()
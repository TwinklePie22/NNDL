import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Load datasets and preprocess
def load_data():
    diabetes = load_diabetes()
    cancer = load_breast_cancer()
    sonar = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header=None)

    X_diabetes = diabetes.data
    y_diabetes = (diabetes.target > diabetes.target.mean()).astype(int)
    X_cancer = cancer.data
    y_cancer = cancer.target
    X_sonar = sonar.iloc[:, :-1].values
    y_sonar = sonar.iloc[:, -1].map({'R': 0, 'M': 1}).values

    return (X_diabetes, y_diabetes), (X_cancer, y_cancer), (X_sonar, y_sonar)

# Create and train model
def create_and_train_model(X, y, activation):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation=activation),
        layers.Dense(32, activation=activation),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    return accuracy_score(y_test, y_pred)

# Evaluate model with a single activation function
def evaluate_models(activation):
    datasets = load_data()
    for dataset, name in zip(datasets, ['Diabetes', 'Cancer', 'Sonar']):
        X, y = dataset
        accuracy = create_and_train_model(X, y, activation)
        print(f"{name} Dataset Accuracy: {accuracy:.2f}")

# Choose a single activation function
activation_function = 'relu'  # Example: Use 'relu', 'sigmoid', or 'tanh'
evaluate_models(activation_function)

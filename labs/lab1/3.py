import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data():
    data = pd.read_csv("src/glass.csv")
    data.drop(columns=["Id"], inplace=True)
    X = data.drop(columns=["Type"])
    y = data["Type"]
    return X, y


def evaluate_dataset(X, y, neighbors):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    errors = []
    for k in neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        error = 1 - accuracy_score(y_test, y_pred)
        errors.append(error)
    return errors


def plot_error_neighbors(neighbors, errors):
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, errors, linestyle='-')
    plt.title('Dependence of Classification Error on Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Classification Error')
    plt.xticks(np.arange(0, 50, step=10))
    plt.grid(True)
    plt.show()


def calculate_metrics(X, y, distance_metrics):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    accuracy_results = {}
    for metric in distance_metrics:
        knn_model = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[metric] = accuracy
    for metric, accuracy in accuracy_results.items():
        print(f"Accuracy with {metric} distance metric: {accuracy}")


def predict_type(X, y, neighbors, instance):
    predicted_types = []
    for k in neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
        knn_model.fit(X, y)
        predicted_type = knn_model.predict(instance)
        predicted_types.append(predicted_type[0])
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, predicted_types, linestyle='-')
    plt.title('Predicted Glass Type vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Predicted Glass Type')
    plt.xticks(np.arange(0, 50, step=10))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, y = load_data()
    neighbors = range(1, 50)
    # Task A
    errors = evaluate_dataset(X, y, neighbors)
    plot_error_neighbors(neighbors, errors)

    # Task B
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    calculate_metrics(X, y, distance_metrics)

    # Task C
    instance = pd.DataFrame([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]], columns=X.columns)
    predict_type(X, y, neighbors, instance)

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter='\t')
    test_data = pd.read_csv(test_file, delimiter='\t')

    X_train = train_data.drop(columns=['SeriousDlqin2yrs'])
    y_train = train_data['SeriousDlqin2yrs']
    X_test = test_data.drop(columns=['SeriousDlqin2yrs'])
    y_test = test_data['SeriousDlqin2yrs']

    return X_train, y_train, X_test, y_test


def evaluate_dataset(method, X_train, y_train, X_test, y_test):
    model = None
    if method == "Gaussian":
        model = GaussianNB()
    elif method == "Neighbors":
        model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix


def plot_matrix(method, conf_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    title = 'Confusion Matrix ' + method
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center',
                     color='black')
    plt.show()


def evaluate_dataset_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': range(1, 20),
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    optimal_clf = grid_search.best_estimator_
    print("Значение max_depth в наилучшей модели:", optimal_clf.max_depth)

    y_pred = optimal_clf.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred)
    conf_matrix_tree = confusion_matrix(y_test, y_pred)
    return accuracy_tree, conf_matrix_tree


if __name__ == "__main__":
    train_file = 'src/bank_scoring_train.csv'
    test_file = 'src/bank_scoring_test.csv'

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    accuracy_Gauss, conf_matrix_Gauss = evaluate_dataset("Gaussian", X_train, y_train, X_test, y_test)
    print("Accuracy with Gaussian =", accuracy_Gauss)
    plot_matrix("Gaussian", conf_matrix_Gauss)

    accuracy_tree, conf_matrix_tree = evaluate_dataset_tree(X_train, y_train, X_test, y_test)
    print("Accuracy with tree =", accuracy_tree)
    plot_matrix("tree", conf_matrix_tree)

    # accuracy_neighbors, conf_matrix_neighbors = evaluate_dataset("Neighbors", X_train, y_train, X_test, y_test)
    # print("Accuracy with neighbors =", accuracy_neighbors)
    # plot_matrix("Neighbors", conf_matrix_neighbors)

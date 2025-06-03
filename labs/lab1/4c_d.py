import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter='\t')
    test_data = pd.read_csv(test_file, delimiter='\t')

    X_train = train_data.drop('Colors', axis=1)
    y_train = train_data['Colors']
    X_test = test_data.drop('Colors', axis=1)
    y_test = test_data['Colors']

    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train, kernel, gamma, degree=None):
    if kernel == 'poly':
        svm_classifier = SVC(kernel=kernel, degree=degree, gamma=gamma)
    else:
        svm_classifier = SVC(kernel=kernel, gamma=gamma)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def visualize_decision_boundary_linear(svm_classifier, X_train, y_train, kernel, gamma, degree=3):
    disp = DecisionBoundaryDisplay.from_estimator(
        svm_classifier,
        X_train,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        xlabel='X1',
        ylabel='X2',
    )

    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=20, edgecolors="k")
    title = f'SVC with {kernel} kernel, {gamma} gamma'
    if kernel == 'poly':
        title = f'SVC with {kernel} kernel {degree} degree, {gamma} gamma'
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    train_file = 'src/svmdata_e.txt'
    test_file = 'src/svmdata_e_test.txt'

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)
    kernels = ['poly', 'rbf', 'sigmoid']
    poly_degrees = [1, 2, 3, 4, 5]
    gammas = [1, 50, 100]

    for kernel in kernels:
        for gamma in gammas:
            if kernel == 'poly':
                for degree in poly_degrees:
                    if degree == 5 and gamma == 50:
                        continue
                    svm_classifier = train_svm(X_train, y_train, kernel, gamma, degree)
                    visualize_decision_boundary_linear(svm_classifier, X_train, y_train, kernel, gamma, degree)
            else:
                svm_classifier = train_svm(X_train, y_train, kernel, gamma)
                visualize_decision_boundary_linear(svm_classifier, X_train, y_train, kernel, gamma)

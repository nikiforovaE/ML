import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(dataset):
    if dataset == "tic_tac_toe":
        data = pd.read_csv("src/tic_tac_toe.txt", header=None)
        data_encoded = pd.get_dummies(data.iloc[:, :-1])
        X = data_encoded.values
        y = data[9]
    else:
        data = pd.read_csv("src/spam.csv")
        X = data.iloc[:, 1:-1]
        y = data.iloc[:, -1]
    return X, y


def evaluate_dataset(dataset):
    X, y = load_data(dataset)
    test_accs = []
    ratios = np.arange(0.1, 1.0, 0.1)
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
        if dataset == "tic_tac_toe":
            model = CategoricalNB()
            model.fit(X_train, y_train)
            test_acc = accuracy_score(y_test, model.predict(X_test))
            test_accs.append(test_acc)
        else:
            model = GaussianNB()
            model.fit(X_train, y_train)
            test_acc = accuracy_score(y_test, model.predict(X_test))
            test_accs.append(test_acc)
    return ratios, test_accs


def plot_results(ratios, test_accs, title):
    plt.xlim(0, max(ratios) + 0.1)
    plt.ylim(0, max(test_accs) + 0.1)
    plt.grid(True)
    plt.plot(ratios, test_accs, label='Test Accuracy')
    plt.xlabel('Train Size Ratio')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tic_tac_toe_ratios, tic_tac_toe_test_accs = evaluate_dataset("tic_tac_toe")
    plot_results(tic_tac_toe_ratios, tic_tac_toe_test_accs, "tic_tac_toe")

    spam_ratios, spam_test_accs = evaluate_dataset("spam")
    plot_results(spam_ratios, spam_test_accs, "spam")

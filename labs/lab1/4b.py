import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter='\t')
    test_data = pd.read_csv(test_file, delimiter='\t')

    X_train = train_data.drop('Colors', axis=1)
    y_train = train_data['Colors']
    X_test = test_data.drop('Colors', axis=1)
    y_test = test_data['Colors']

    return X_train, y_train, X_test, y_test

def train_svm_models(X_train, y_train, X_test, y_test):
    train_accuracy_list = []
    test_accuracy_list = []
    C_values = []

    C_range = range(1, 1000, 1)

    for C in C_range:
        svm_model = SVC(kernel='linear', C=C)
        svm_model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
        train_accuracy_list.append(train_accuracy)

        test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
        test_accuracy_list.append(test_accuracy)

        C_values.append(C)

    return C_values, train_accuracy_list, test_accuracy_list

def plot_accuracy_vs_C(C_values, train_accuracy_list, test_accuracy_list):
    plt.plot(C_values, train_accuracy_list, label='Обучающий набор данных')
    plt.plot(C_values, test_accuracy_list, label='Тестовый набор данных')
    plt.xlabel('Значение C')
    plt.ylabel('Точность')
    plt.title('Зависимость точности от значения C')
    plt.xticks(np.arange(0, 1000, step=30))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_file = 'src/svmdata_b.txt'
    test_file = 'src/svmdata_b_test.txt'

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    C_values, train_accuracy_list, test_accuracy_list = train_svm_models(X_train, y_train, X_test, y_test)
    plot_accuracy_vs_C(C_values, train_accuracy_list, test_accuracy_list)

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
from sklearn.svm import SVC


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter='\t')
    test_data = pd.read_csv(test_file, delimiter='\t')

    X_train = train_data.drop('Color', axis=1)
    y_train = train_data['Color']
    X_test = test_data.drop('Color', axis=1)
    y_test = test_data['Color']

    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def visualize_decision_boundary(svm_classifier, X_train, y_train):
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
    plt.title('SVC with linear kernel')
    plt.show()


def compute_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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


if __name__ == "__main__":
    train_file = 'src/svmdata_a.txt'
    test_file = 'src/svmdata_a_test.txt'

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    svm_classifier = train_svm(X_train, y_train)
    visualize_decision_boundary(svm_classifier, X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    n_support_vectors = len(svm_classifier.support_vectors_)
    print("Number of support vectors:", n_support_vectors)

    compute_confusion_matrix(y_test, y_pred)

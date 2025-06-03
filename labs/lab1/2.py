import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt


def plot_points(data_1, data_2):
    # Визуализация данных
    plt.figure(figsize=(10, 6))
    plt.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class -1')
    plt.scatter(data_2[:, 0], data_2[:, 1], c='b', label='Class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Data')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return X_test, y_test, y_pred, y_pred_proba


def plot_matrix():
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['-1', '1'])
    plt.yticks([0, 1], ['-1', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center',
                     color='black')
    plt.show()


def plot_ROC(y_test, y_pred_proba):
    # рассчет ROC
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # построение ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_PR(y_test, y_pred_proba):
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)

    # построение PR-кривых
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Naive Bayes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    # Параметры класса -1
    mean_1 = [13, 20]
    covariance_1 = [[16, 0], [0, 16]]
    num_samples_1 = 60

    # Параметры класса 1
    mean_2 = [6, 8]
    covariance_2 = [[1, 0], [0, 1]]
    num_samples_2 = 40

    data_1 = np.random.multivariate_normal(mean_1, covariance_1, num_samples_1)
    data_2 = np.random.multivariate_normal(mean_2, covariance_2, num_samples_2)

    X = np.vstack((data_1, data_2))
    y = np.hstack((np.full(num_samples_1, -1), np.full(num_samples_2, 1)))

    plot_points(data_1, data_2)

    X_test, y_test, y_pred, y_pred_proba = evaluate_dataset(X, y)

    # Оценка качества классификации
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    plot_matrix()
    plot_ROC(y_test, y_pred_proba)
    plot_PR(y_test, y_pred_proba)


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('src/spam7.csv')

X = data.drop(columns=['yesno'])
y = data['yesno']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
default_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", default_accuracy)
default_depth = clf.tree_.max_depth
print("Depth: ", default_depth)

plt.figure(figsize=(13, 6))
plot_tree(clf, feature_names=X.columns, filled=True)
plt.show()

# Определение диапазона параметров для подбора
param_grid = {
    'max_depth': range(1, 15),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 10)
}

clf = DecisionTreeClassifier(random_state=42)

# Поиск наилучших параметров модели
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Наилучшие параметры:", grid_search.best_params_)

# Построение оптимальной модели
optimal_clf = grid_search.best_estimator_

y_pred = optimal_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Точность наилучшей модели:", accuracy)

plt.figure(figsize=(13, 6))
plot_tree(optimal_clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()

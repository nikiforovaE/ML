import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = pd.read_csv('src/glass.csv')
X = data.drop(columns=['Id', 'Type'])
y = data['Type']

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

results = []
for max_depth in range(1, 21):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((max_depth, accuracy))


min_depth = default_depth
for max_depth, accuracy in results:
    if max_depth < default_depth:
        if accuracy > default_accuracy:
            print(max_depth, accuracy)

    else:
        break

results_df = pd.DataFrame(results, columns=['Max Depth', 'Accuracy'])
print(results_df)

clf = DecisionTreeClassifier(max_depth=8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
default_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", default_accuracy)
print("Depth: ", 8)

plt.figure(figsize=(13, 6))
plot_tree(clf, feature_names=X.columns, filled=True)
plt.show()
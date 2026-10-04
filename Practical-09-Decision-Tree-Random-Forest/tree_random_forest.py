import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
forest = RandomForestClassifier(n_estimators=100, random_state=42)

tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

tree_accuracy = accuracy_score(
    y_test, tree.predict(X_test)
)

forest_accuracy = accuracy_score(
    y_test, forest.predict(X_test)
)

print("Decision Tree:", round(tree_accuracy, 2))
print("Random Forest:", round(forest_accuracy, 2))

plt.bar(
    ["Decision Tree", "Random Forest"],
    [tree_accuracy, forest_accuracy]
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Tree vs Random Forest")
plt.show()

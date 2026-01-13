import pandas as tehreem
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

print("\n--- Q2: Implement Decision Tree Classifier on a Small Dataset. Build and visualize a simple decision tree. ---\n")
# ------------------- Q2 -------------------
df = tehreem.read_csv("DataFile_A3\DataFile3.csv")
print("Original Dataset:\n", df)

le = LabelEncoder()
df['Study Hours'] = le.fit_transform(df['Study Hours'])
df['Attendance'] = le.fit_transform(df['Attendance'])
df['Result'] = le.fit_transform(df['Result'])

print("\nEncoded Dataset:\n", df)

X = df[['Study Hours', 'Attendance']]
y = df['Result']

clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)

plt.figure(figsize=(6,4))
plot_tree(clf, feature_names=['Study Hours', 'Attendance'], 
          class_names=['Fail', 'Pass'], filled=True, rounded=True)
plt.show()

sample = [[1, 0]]  # Low=1, Good=0
prediction = clf.predict(sample)

print("\nPrediction for Study Hours=Low and Attendance=Good:", 
      "Pass" if prediction[0]==1 else "Fail")

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q3: Decision Tree Classifier on Iris Dataset. Objective: Apply decision trees to a real dataset. ---\n")

# ------------------- Q3 -------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", round(accuracy, 3))

plt.figure(figsize=(12,6))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.show()

root_feature_index = clf.tree_.feature[0]
print("Feature providing most information gain at the root:",
      iris.feature_names[root_feature_index])


print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q4: MNIST digit dataset (available via Keras / sklearn.datasets) as a baseline ---\n")

# ------------------- Q4 -------------------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X = digits.data
y = digits.target

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy on MNIST test set:", round(acc, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree on MNIST")
plt.show()

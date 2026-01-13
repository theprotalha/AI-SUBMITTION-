
# print("\n--- Q1: Classify flower species using Random Forest. ---\n")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# iris = pd.read_csv('DataFile_A4/DataFile4-Q1.csv')

# X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y = iris['Species']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("____________________________________________________________________________________________________________________________________________")

# print("\n--- Q2: Use SVM on Breast Cancer Dataset and Classify tumors as malignant or benign. ---\n")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix

# data = pd.read_csv(r'DataFile_A4\DataFile4-Q2.csv')

# X = data[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
# y = data['diagnosis'].replace({'B': 0, 'M': 1})

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# svm_model = SVC(kernel='linear', random_state=42)
# svm_model.fit(X_train, y_train)

# y_pred = svm_model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# results = pd.DataFrame({
#     'Actual Diagnosis': ['Benign' if val == 0 else 'Malignant' for val in y_test],
#     'Predicted Diagnosis': ['Benign' if val == 0 else 'Malignant' for val in y_pred]
# })

# print("\nPredicted Tumor Diagnosis (10 random samples):")
# print(results.sample(10))

# print("____________________________________________________________________________________________________________________________________________")

# print("\n--- Q3: Use Random Forest on CSV Dataset (Custom) : Predict student pass/fail based on study hours and scores. ---\n")
# # ------------------- Q3 -------------------
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# data = pd.read_csv("DataFile_A4\DataFile4-Q3.csv")

# data['result'] = data['result'].map({'Pass': 1, 'Fail': 0})

# X = data[['study_hours', 'attendance', 'marks']]
# y = data['result']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# importance = pd.Series(rf.feature_importances_, index=X.columns)
# print("\nFeature Importance:\n", importance)

# print("____________________________________________________________________________________________________________________________________________")

print('--- Q4: Use SVM on Digits Dataset to identify Handwritten Digits. ---')

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
digits = load_digits()
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    digits.data, digits.target, digits.images, test_size=0.3, random_state=42
)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
classified = (y_test == y_pred) 
classified_indices = np.where(classified)[0]
plt.figure(figsize=(8,4))
plt.suptitle("First six Correctly Predicted Digits")
for i, img_idx in enumerate(classified_indices[:6]):
    plt.subplot(2,3,i+1)
    plt.imshow(images_test[img_idx], cmap='Reds')
    plt.title(f"Pred:{y_pred[img_idx]}, True:{y_test[img_idx]}")
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("____________________________________________________________________________________________________________________________________________")

# print("\n--- Q5: Compare Random Forest and SVM on Custom Dataset. ---\n")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# data = pd.read_csv(r'DataFile_A4\DataFile4-Q5.csv')

# X = data[['H', 'S', 'V', 'L']].fillna(data[['H', 'S', 'V', 'L']].mean())
# y = data['group'].fillna(data['group'].mode()[0])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# svm = SVC(kernel='rbf', gamma='scale')
# svm.fit(X_train, y_train)
# svm_pred = svm.predict(X_test)

# rf_acc = accuracy_score(y_test, rf_pred)
# svm_acc = accuracy_score(y_test, svm_pred)

# print("Random Forest Accuracy:", rf_acc)
# print("SVM Accuracy:", svm_acc)

# if rf_acc > svm_acc:
#     print("\nConclusion: Random Forest performed better.")
# elif svm_acc > rf_acc:
#     print("\nConclusion: SVM performed better.")
# else:
#     print("\nConclusion: Both performed equally well.")

# print("____________________________________________________________________________________________________________________________________________")

import pandas as tehreem
import numpy as np
import matplotlib.pyplot as syed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score

print("\n--- Q1: Multiple Linear Regression – House Price Prediction ---\n")
# ------------------- Q1 -------------------
data1 = tehreem.read_csv("DataFile_A2\DataFile2-Q1.csv")

X1 = data1[['Size (sqft)', 'Bedrooms', 'Age (Years)']]
y1 = data1['Price']

model1 = LinearRegression()
model1.fit(X1, y1)

house = tehreem.DataFrame([[2000, 3, 10]], columns=['Size (sqft)', 'Bedrooms', 'Age (Years)'])
pred_price = model1.predict(house)

print("1. Fit a multiple linear regression model.\n")
print("2. Predicted Price for (2000 sqft, 3 bedrooms, 10 years):", pred_price[0], "\n")
print("3. Intercept:", model1.intercept_, "\n")
print("   Coefficients:", model1.coef_, "\n")

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q2: Multiple Linear Regression – Student Performance ---\n")

# ------------------- Q2 -------------------
data2 = tehreem.read_csv("DataFile_A2\DataFile2-Q2.csv")

X2 = data2[['Hours Study', 'Hours Sleep', 'Attendance (%)']]
y2 = data2['Marks']

model2 = LinearRegression()
model2.fit(X2, y2)

y2_pred = model2.predict(X2)

print("1. Regression model trained successfully.\n")
print("2. Plotting Actual vs Predicted Marks...\n")

syed.scatter(y2, y2_pred, color='red')
syed.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'g--')
syed.xlabel("Actual Marks")
syed.ylabel("Predicted Marks")
syed.title("Q2: Actual vs Predicted Marks")
syed.show()

print("3. Model Performance:\n")
print("   R² Score:", r2_score(y2, y2_pred), "\n")
print("   Mean Squared Error (MSE):", mean_squared_error(y2, y2_pred), "\n")

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q3: Logistic Regression – Pass/Fail Classification ---\n")

# ------------------- Q3 -------------------
data3 = tehreem.read_csv("DataFile_A2\DataFile2-Q3.csv")

X3 = data3[['Hours Study', 'Hours Sleep']]
y3 = data3['Pass']

model3 = LogisticRegression()
model3.fit(X3, y3)

student = tehreem.DataFrame([[30, 6]], columns=['Hours Study', 'Hours Sleep'])
prob_pass = model3.predict_proba(student)[0][1]

print("1. Logistic Regression model trained successfully.\n")
print("2. Probability of Passing (30 study hrs, 6 sleep hrs):", prob_pass, "\n")
print("3. Plotting dataset points (Pass = 1, Fail = 0)...\n")

syed.scatter(data3['Hours Study'], data3['Hours Sleep'], c=data3['Pass'], cmap='bwr')
syed.xlabel("Hours Study")
syed.ylabel("Hours Sleep")
syed.title("Q3: Pass/Fail Classification")
syed.show()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q4. Logistic Regression – Diabetes Prediction (Binary Classification) ---\n")

# ------------------- Q4 -------------------
data4 = tehreem.read_csv("DataFile_A2\DataFile2-Q4.csv")

X4 = data4[['BMI', 'Age', 'Glucose']]
y4 = data4['Diabetic']

model4 = LogisticRegression()
model4.fit(X4, y4)

y4_pred = model4.predict(X4)

print("1. Logistic Regression model trained successfully.\n")

acc = accuracy_score(y4, y4_pred)
prec = precision_score(y4, y4_pred)
rec = recall_score(y4, y4_pred)

print("2. Model Performance:\n")
print("   Accuracy :", acc)
print("   Precision:", prec)
print("   Recall   :", rec, "\n")

patient = tehreem.DataFrame([[28, 45, 150]], columns=['BMI', 'Age', 'Glucose'])
prediction = model4.predict(patient)[0]

print("3. Prediction for patient (BMI=28, Age=45, Glucose=150):")
print("   Diabetic\n" if prediction == 1 else "   Not Diabetic\n")

print("4. Plotting dataset points (Diabetic=1, Not=0)...\n")

syed.scatter(data4['Glucose'], data4['BMI'], c=data4['Diabetic'], cmap='bwr')
syed.xlabel("Glucose Level")
syed.ylabel("BMI")
syed.title("Q4: Diabetes Prediction (Binary Classification)")
syed.show()

print("____________________________________________________________________________________________________________________________________________")

print("\n--- Q5. Comparison – Linear vs Logistic Regression ---\n")

# ------------------- Q5 -------------------
data5 = tehreem.read_csv("DataFile_A2\DataFile2-Q5.csv")

X5 = data5[['Hours Study']]
y5_score = data5['Exam Score']
y5_pass = data5['Pass']

# ---------- Linear Regression ----------
lin_model = LinearRegression()
lin_model.fit(X5, y5_score)
y5_score_pred = lin_model.predict(X5)

print("1. Linear Regression model trained to predict exam scores.\n")

syed.scatter(X5, y5_score, color='blue', label="Actual Score")
syed.plot(X5, y5_score_pred, color='red', label="Predicted (Linear)")
syed.xlabel("Hours Study")
syed.ylabel("Exam Score")
syed.title("Q5: Linear Regression - Exam Score Prediction")
syed.legend()
syed.show()

# Calculate MSE & RMSE for Linear Regression
mse_linear = mean_squared_error(y5_score, y5_score_pred)
rmse_linear = np.sqrt(mse_linear)

print("Linear Regression Performance:")
print("   Mean Squared Error (MSE):", mse_linear)
print("   Root Mean Squared Error (RMSE):", rmse_linear, "\n")

# ---------- Logistic Regression ----------
log_model = LogisticRegression()
log_model.fit(X5, y5_pass)
y5_pass_pred = log_model.predict(X5)

print("2. Logistic Regression model trained to predict pass/fail.\n")

syed.scatter(X5, y5_pass, color='blue', label="Actual Pass/Fail")
syed.scatter(X5, y5_pass_pred, color='red', marker='x', label="Predicted Pass/Fail")
syed.xlabel("Hours Study")
syed.ylabel("Pass/Fail")
syed.title("Q5: Logistic Regression - Pass/Fail Classification")
syed.legend()
syed.show()

# Calculate Accuracy for Logistic Regression
acc_logistic = accuracy_score(y5_pass, y5_pass_pred)

print("Logistic Regression Performance:")
print("   Accuracy:", acc_logistic, "\n")

# ---------- Comparison ----------
print("3. Comparison Summary:\n")
print("   ➤ Linear Regression:")
print("       - Predicts continuous values (like marks)")
print("       - MSE:", mse_linear)
print("       - RMSE:", rmse_linear)
print("   ➤ Logistic Regression:")
print("       - Predicts binary outcomes (Pass/Fail)")
print("       - Accuracy:", acc_logistic)
print("\nConclusion:")
print("   - For score prediction → Linear Regression is better (lower MSE/RMSE means more accurate).")
print("   - For pass/fail prediction → Logistic Regression is better (higher Accuracy means more correct classification).")

print("____________________________________________________________________________________________________________________________________________")
print("\n\n")
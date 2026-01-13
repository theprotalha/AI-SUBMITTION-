print("\n--- Q1: Create Dataset ---\n")
print('\n I created the dataset in the excel file and saved that file with ".csv" extension \n')
print("\n--- Q2: Upload dataset in python ---\n")

import pandas as naqvi
tehreem_data = naqvi.read_csv("DataFile_A1\DataFile1.csv")
print(tehreem_data)

print("\n--- Q3: Dataset Information ---")
print("\n 1. Structure of dataset \n")
print(tehreem_data.info())

print("\n 2. Summary Statistics \n")
print(tehreem_data.describe())

print("\n 3. Mean of Math Marks data", tehreem_data["Marks_Math"].mean(), "\n")

print("\n 4. Maximum Science Marks data", tehreem_data["Marks_Science"].max(), "\n", flush=True)

print("\n--- Q4: Data Analysis ---\n")

math_score_count = (tehreem_data["Marks_Math"]>50).sum()
print("\n 1. Students with Math Marks > 50:", math_score_count)

highest_science_score = tehreem_data[tehreem_data["Marks_Science"] == tehreem_data["Marks_Science"].max()]
print("\n 2.Student with highest science marks:\n", "\n", highest_science_score, "\n")

print("\n 3. Correlation between Math & Science Marks:\n", "\n", tehreem_data[["Marks_Math", "Marks_Science"]].corr(), "\n")

print("\n--- Q5: Data Visualization ---\n")

import matplotlib.pyplot as syed

syed.bar(tehreem_data["Student_ID"], tehreem_data["Marks_Math"])
syed.title("Math Marks of Students")
syed.xlabel("Student_ID")
syed.ylabel("Marks_Math")
syed.show()

syed.hist(tehreem_data["Age"], bins=5, color="lavender", edgecolor="hotpink")
syed.title("Distribution of Age")
syed.xlabel("Age")
syed.ylabel("Count")
syed.show()

syed.scatter(tehreem_data["Marks_Math"], tehreem_data["Marks_Science"], color="red")
syed.title("Math vs Science Marks")
syed.xlabel("Marks_Math")
syed.ylabel("Marks_Science")
syed.show()


"""
Practical 4 - Exploratory Data Analysis (EDA) on two real datasets:
the Titanic passenger manifest and the real UCI Student Performance dataset
(395 real Portuguese secondary school students).
Run: python eda_demo.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("images", exist_ok=True)


def eda_report(df, name):
    """A real, reusable EDA summary: shape, types, missing values, and
    basic real statistics -- the first things any real data scientist
    checks before doing anything else with a new dataset."""
    print(f"\n### EDA: {name} ###")
    print(f"Real shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nReal column types:\n{df.dtypes.value_counts()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nReal missing values:\n{missing}")
    else:
        print("\nNo real missing values.")
    print(f"\nReal numeric summary:\n{df.describe().round(2)}")


if __name__ == "__main__":
    titanic = pd.read_csv("../datasets/titanic.csv")
    eda_report(titanic, "Titanic (891 real passengers)")

    student = pd.read_csv("../datasets/student-mat.csv")
    eda_report(student, "Student Performance (395 real students)")

    # Real, focused comparison: what predicts the real final grade (G3)?
    print("\n### REAL FOCUSED QUESTION: WHAT PREDICTS STUDENT G3 (FINAL GRADE)? ###")
    numeric_student = student.select_dtypes(include="number")
    corr_with_g3 = numeric_student.corr()["G3"].drop("G3").sort_values(key=abs, ascending=False)
    print("Real top 5 correlates of final grade (G3):")
    print(corr_with_g3.head(5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(titanic["Age"].dropna(), bins=25, color="#2563eb")
    axes[0].set_title("Real Titanic Age Distribution")
    axes[0].set_xlabel("Age")

    axes[1].hist(student["G3"], bins=20, color="#16a34a")
    axes[1].set_title("Real Student Final Grade (G3) Distribution")
    axes[1].set_xlabel("G3 (0-20 scale)")
    plt.tight_layout()
    plt.savefig("images/01_distributions_comparison.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(student["studytime"], student["G3"], alpha=0.5, color="#f97316")
    plt.xlabel("Real weekly study time (1=low, 4=high)")
    plt.ylabel("Real final grade (G3)")
    plt.title("Real Study Time vs Final Grade")
    plt.tight_layout()
    plt.savefig("images/02_studytime_vs_grade.png")
    plt.close()

    print("\nDone. 2 real charts saved in images/")

"""
Practical 2 - Data manipulation using NumPy and Pandas, on the real
Titanic passenger dataset (891 real passengers, 1912 sinking).
Run: python numpy_pandas_demo.py
"""
import numpy as np
import pandas as pd

df = pd.read_csv("../datasets/titanic.csv")

if __name__ == "__main__":
    print("### LOADING THE REAL DATASET ###")
    print(f"Real shape: {df.shape[0]} real passengers, {df.shape[1]} real columns")
    print(df.head(3))

    print("\n### NUMPY: REAL ARRAY OPERATIONS ON FARE ###")
    fares = df["Fare"].values  # a real numpy array, pulled straight out of the DataFrame
    print(f"Real mean fare: £{np.mean(fares):.2f}")
    print(f"Real median fare: £{np.median(fares):.2f}")
    print(f"Real standard deviation: £{np.std(fares):.2f}")
    print(f"Real max fare: £{np.max(fares):.2f} (paid by passenger index {np.argmax(fares)})")

    print("\n### PANDAS: REAL FILTERING AND GROUPING ###")
    survivors = df[df["Survived"] == 1]
    print(f"Real survivors: {len(survivors)} out of {len(df)} "
          f"({len(survivors) / len(df):.1%} real survival rate)")

    by_class = df.groupby("Pclass")["Survived"].mean()
    print("\nReal survival rate by real passenger class:")
    print(by_class)

    print("\n### PANDAS: REAL SORTING AND SELECTION ###")
    oldest_5 = df.nlargest(5, "Age")[["Name", "Age", "Pclass", "Survived"]]
    print("\nReal 5 oldest passengers on board:")
    print(oldest_5.to_string(index=False))

    print("\n### NUMPY + PANDAS TOGETHER: REAL CONDITIONAL LOGIC ###")
    df["AgeGroup"] = np.where(df["Age"] < 18, "Child", np.where(df["Age"] < 60, "Adult", "Senior"))
    age_group_survival = df.groupby("AgeGroup")["Survived"].mean()
    print("\nReal survival rate by real age group:")
    print(age_group_survival)

    print("\nDone.")

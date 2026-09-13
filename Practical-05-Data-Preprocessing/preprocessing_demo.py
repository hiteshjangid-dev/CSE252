"""
Practical 5 - Data preprocessing on the real Titanic dataset: handling
missing values, encoding categorical features, and feature scaling --
the 3 real steps almost every ML pipeline needs before training.
Run: python preprocessing_demo.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("../datasets/titanic.csv")

if __name__ == "__main__":
    print("### STEP 1: REAL MISSING VALUES, BEFORE ###")
    print(df[["Age", "Cabin", "Embarked"]].isnull().sum())

    # Real, standard approach: fill Age with the real median (robust to outliers),
    # fill Embarked with the real most common port, drop Cabin (too sparse to use)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])

    print("\n### STEP 1: REAL MISSING VALUES, AFTER ###")
    print(df[["Age", "Embarked"]].isnull().sum())
    print("Cabin column dropped (687/891 real values were missing -- too sparse to use)")

    print("\n### STEP 2: REAL CATEGORICAL ENCODING ###")
    print("Real 'Sex' values before encoding:", df["Sex"].unique())
    encoder = LabelEncoder()
    df["Sex_encoded"] = encoder.fit_transform(df["Sex"])
    print("Real 'Sex' values after encoding:", df["Sex_encoded"].unique(),
          "(mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))), ")")

    print("\nReal 'Embarked' values before encoding:", df["Embarked"].unique())
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")
    embarked_cols = [c for c in df.columns if c.startswith("Embarked_")]
    print("Real one-hot columns created:", embarked_cols)

    print("\n### STEP 3: REAL FEATURE SCALING ###")
    print("Real Age and Fare BEFORE scaling:")
    print(df[["Age", "Fare"]].describe().round(2).loc[["mean", "std", "min", "max"]])

    scaler = StandardScaler()
    df[["Age_scaled", "Fare_scaled"]] = scaler.fit_transform(df[["Age", "Fare"]])
    print("\nReal Age and Fare AFTER scaling (mean should be ~0, std ~1):")
    print(df[["Age_scaled", "Fare_scaled"]].describe().round(2).loc[["mean", "std", "min", "max"]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["Fare"], bins=30, color="#dc2626")
    axes[0].set_title("Real Fare BEFORE Scaling")
    axes[1].hist(df["Fare_scaled"], bins=30, color="#16a34a")
    axes[1].set_title("Real Fare AFTER Scaling")
    plt.tight_layout()
    plt.savefig("images/01_before_after_scaling.png")
    plt.close()

    print("\nReal final preprocessed shape:", df.shape)
    print("Done. Chart saved in images/")

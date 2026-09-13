"""
Practical 3 - Data visualization using Matplotlib, on the real Titanic
passenger dataset.
Run: python matplotlib_demo.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../datasets/titanic.csv")
os.makedirs("images", exist_ok=True)

if __name__ == "__main__":
    print(f"Real dataset: {len(df)} real passengers")

    # Chart 1: real age distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Age"].dropna(), bins=30, color="#2563eb", edgecolor="white")
    plt.xlabel("Age")
    plt.ylabel("Number of real passengers")
    plt.title("Real Age Distribution of Titanic Passengers")
    plt.tight_layout()
    plt.savefig("images/01_age_distribution.png")
    plt.close()
    print("Saved: real age distribution histogram")

    # Chart 2: real survival counts by class
    survival_by_class = df.groupby(["Pclass", "Survived"]).size().unstack()
    survival_by_class.plot(kind="bar", figsize=(8, 5), color=["#dc2626", "#16a34a"])
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of real passengers")
    plt.title("Real Survival Counts by Passenger Class")
    plt.legend(["Did not survive", "Survived"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("images/02_survival_by_class.png")
    plt.close()
    print("Saved: real survival by class bar chart")

    # Chart 3: real fare vs age scatter, colored by survival
    plt.figure(figsize=(8, 6))
    colors = df["Survived"].map({0: "#dc2626", 1: "#16a34a"})
    plt.scatter(df["Age"], df["Fare"], c=colors, alpha=0.6, s=30)
    plt.xlabel("Age")
    plt.ylabel("Fare (£)")
    plt.title("Real Age vs Fare, Colored by Real Survival Outcome")
    plt.tight_layout()
    plt.savefig("images/03_age_vs_fare.png")
    plt.close()
    print("Saved: real age vs fare scatter plot")

    # Chart 4: real correlation heatmap-style summary
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.title("Real Correlation Between Numeric Titanic Features")
    plt.tight_layout()
    plt.savefig("images/04_correlation_heatmap.png")
    plt.close()
    print("Saved: real correlation heatmap")

    strongest = corr["Survived"].drop("Survived").abs().idxmax()
    print(f"\nReal strongest correlate of survival: {strongest} "
          f"(correlation = {corr['Survived'][strongest]:.2f})")

    print("\nDone. 4 real charts saved in images/")

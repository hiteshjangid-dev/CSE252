import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
df = pd.read_csv("datasets/titanic.csv")

print("Original Shape:", df.shape)

print("\nMissing Values Before:")
print(df.isnull().sum())


# Handle missing values
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df = df.drop(columns=["Cabin"])


# Encode Gender
encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(
    df["Gender"]
)


# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)


# Separate features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]


# Remove ID and text columns
X = X.drop(
    columns=["PassengerId", "Name", "Ticket"]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale numerical features
scaler = StandardScaler()

X_train[["Age", "Fare"]] = scaler.fit_transform(
    X_train[["Age", "Fare"]]
)

X_test[["Age", "Fare"]] = scaler.transform(
    X_test[["Age", "Fare"]]
)


print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nMissing Values After:")
print(X_train.isnull().sum())

print("\nScaled Features:")
print(
    X_train[["Age", "Fare"]]
    .describe()
    .round(2)
)


# Before and after scaling
original = pd.read_csv(
    "datasets/titanic.csv"
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(
    original["Fare"].dropna(),
    bins=25
)
plt.title("Fare Before Scaling")
plt.xlabel("Fare")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(
    X_train["Fare"],
    bins=25
)
plt.title("Fare After Scaling")
plt.xlabel("Scaled Fare")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "images/01_before_after_scaling.png"
)

plt.show()

print("\nPreprocessing Completed Successfully")

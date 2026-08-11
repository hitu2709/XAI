import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pygam import LogisticGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/diabetes.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)


# ============================================================
# 3. PREPROCESS DATA
# ============================================================

# In this dataset, zero is not meaningful for these columns.
# Replace zero values with NaN.

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_with_zero] = data[columns_with_zero].replace(
    0,
    np.nan
)


# Fill missing values with median

data[columns_with_zero] = data[columns_with_zero].fillna(
    data[columns_with_zero].median()
)


print("\nMissing values after preprocessing:")
print(data.isnull().sum())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("Outcome", axis=1)

y = data["Outcome"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Outcome")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CREATE GAM MODEL
# ============================================================

gam = LogisticGAM(
    s(0) +
    s(1) +
    s(2) +
    s(3) +
    s(4) +
    s(5) +
    s(6) +
    s(7)
)


# ============================================================
# 7. TRAIN GAM MODEL
# ============================================================

gam.fit(
    X_train,
    y_train
)


print("\nGAM model trained successfully!")


# ============================================================
# 8. MODEL SUMMARY
# ============================================================

print("\n================ GAM SUMMARY ================")

print(gam.summary())


# ============================================================
# 9. PREDICTIONS
# ============================================================

y_pred = gam.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================ MODEL PERFORMANCE ================")

print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 11. PDP FOR EVERY FEATURE
# ============================================================

print("\n================ PDP ANALYSIS ================")

feature_names = X.columns


for i, feature in enumerate(feature_names):

    # Generate values for the selected feature
    XX = gam.generate_X_grid(
        term=i
    )


    # Calculate partial dependence
    pdep, confi = gam.partial_dependence(
        term=i,
        X=XX,
        width=0.95
    )


    # Create figure
    plt.figure(
        figsize=(7, 5)
    )


    # Plot partial dependence
    plt.plot(
        XX[:, i],
        pdep,
        linewidth=2,
        label="Partial Dependence"
    )


    # Plot confidence interval
    plt.plot(
        XX[:, i],
        confi,
        linestyle="--",
        label="95% Confidence Interval"
    )


    # Labels
    plt.xlabel(
        feature
    )

    plt.ylabel(
        "Partial Dependence"
    )


    plt.title(
        "Partial Dependence Plot - " + feature
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    filename = (
        "outputs/PDP_" +
        feature +
        ".png"
    )


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        "Saved:",
        filename
    )


    # Close figure
    plt.close()


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n================================================")
print("Experiment 3 completed successfully!")
print("================================================")

print("\nPDP images are saved in:")

print("outputs/")

print("\nGenerated files:")

for feature in feature_names:

    print(
        "PDP_" +
        feature +
        ".png"
    )
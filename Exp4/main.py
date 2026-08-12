import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score

import shap


# ============================================================
# 1. CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. CREATE DATASET
# ============================================================

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "StudyHours": np.random.uniform(1, 15, n),
    "Attendance": np.random.uniform(50, 100, n),
    "PreviousScore": np.random.uniform(40, 95, n),
    "SleepHours": np.random.uniform(4, 9, n),
    "Assignments": np.random.uniform(40, 100, n),
    "ScreenTime": np.random.uniform(1, 10, n)
})

# Create FinalScore with meaningful relationships
noise = np.random.normal(0, 4, n)

data["FinalScore"] = (
    0.30 * data["StudyHours"] +
    0.25 * data["Attendance"] +
    0.35 * data["PreviousScore"] +
    0.15 * data["SleepHours"] +
    0.20 * data["Assignments"] -
    0.20 * data["ScreenTime"] +
    noise
)

# Scale final score to approximately 40-100
data["FinalScore"] = (
    40 + 
    (data["FinalScore"] - data["FinalScore"].min()) /
    (data["FinalScore"].max() - data["FinalScore"].min()) * 60
)

# Save dataset
data.to_csv("dataset/student_performance.csv", index=False)

print("Dataset created successfully!")
print(data.head())


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv("dataset/student_performance.csv")

X = df.drop("FinalScore", axis=1)
y = df["FinalScore"]


# ============================================================
# 4. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n================ MODEL PERFORMANCE ================")
print("R2 Score :", round(r2, 4))
print("MSE      :", round(mse, 4))


# ============================================================
# 7. PERMUTATION FEATURE IMPORTANCE
# ============================================================

print("\n================ PERMUTATION IMPORTANCE ================")

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=10,
    random_state=42
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)


# Plot permutation importance
plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/permutation_importance.png")
plt.show()


# ============================================================
# 8. SHAP ANALYSIS
# ============================================================

print("\n================ SHAP ANALYSIS ================")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()
plt.savefig("outputs/shap_summary.png")
plt.show()


# SHAP Bar Plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("outputs/shap_bar.png")
plt.show()


# ============================================================
# 9. ONE-AT-A-TIME SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(model, X, feature, step=0.10):

    X_varied = X.copy()

    original_value = X[feature].mean()

    values = np.linspace(
        original_value * (1 - step),
        original_value * (1 + step),
        5
    )

    predictions = []

    for value in values:

        X_temp = X.copy()

        X_temp[feature] = value

        preds = model.predict(X_temp)

        predictions.append(np.mean(preds))

    return values, predictions


# Select most important feature
important_feature = importance_df.iloc[0]["Feature"]

print("\n================ OAT SENSITIVITY ================")

values, predictions = sensitivity_analysis(
    model,
    X_test,
    important_feature
)

for value, prediction in zip(values, predictions):

    print(
        important_feature,
        "=",
        round(value, 2),
        "-> Average Prediction =",
        round(prediction, 2)
    )


# Plot OAT sensitivity
plt.figure(figsize=(8, 5))

plt.plot(
    values,
    predictions,
    marker="o"
)

plt.xlabel(important_feature)
plt.ylabel("Average Predicted Final Score")

plt.title(
    "One-at-a-Time Sensitivity Analysis - "
    + important_feature
)

plt.tight_layout()

plt.savefig("outputs/oat_sensitivity.png")

plt.show()


# ============================================================
# 10. PARTIAL DEPENDENCE PLOT
# ============================================================

print("\n================ PDP ANALYSIS ================")

fig, ax = plt.subplots(figsize=(8, 5))

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    [important_feature],
    ax=ax
)

plt.title(
    "Partial Dependence Plot - "
    + important_feature
)

plt.tight_layout()

plt.savefig("outputs/pdp.png")

plt.show()


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n================ FINAL RESULT ================")

print("Most influential feature:",
      important_feature)

print("\nFeature ranking:")

for i, row in importance_df.iterrows():

    print(
        row["Feature"],
        "->",
        round(row["Importance"], 4)
    )

print("\nExperiment completed successfully!")

print("\nGenerated files:")
print("1. outputs/permutation_importance.png")
print("2. outputs/shap_summary.png")
print("3. outputs/shap_bar.png")
print("4. outputs/oat_sensitivity.png")
print("5. outputs/pdp.png")
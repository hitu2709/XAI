
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Setup directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# --------------------------------------------------
# 2. Create synthetic cardiovascular-risk dataset
# --------------------------------------------------

n_samples = 1200

data = pd.DataFrame({
    "age": rng.integers(25, 81, n_samples),
    "bmi": np.round(rng.normal(27, 5, n_samples).clip(16, 45), 1),
    "smoking": rng.integers(0, 2, n_samples),
    "cholesterol": rng.normal(200, 35, n_samples).clip(110, 350),
    "systolic_bp": rng.normal(125, 18, n_samples).clip(85, 210),
    "physical_activity": rng.integers(0, 8, n_samples),
})

# Synthetic risk score used to create demonstration labels.
# This is not a clinically validated risk equation.
risk_score = (
    0.035 * (data["age"] - 45)
    + 0.07 * (data["bmi"] - 25)
    + 0.65 * data["smoking"]
    + 0.012 * (data["cholesterol"] - 180)
    + 0.025 * (data["systolic_bp"] - 120)
    - 0.15 * data["physical_activity"]
    + rng.normal(0, 0.8, n_samples)
)

data["risk"] = (risk_score > 1.2).astype(int)

X = data.drop(columns="risk")
y = data["risk"]

feature_names = X.columns.tolist()

# Save the synthetic dataset
data.to_csv(RESULTS_DIR / "sample_dataset.csv", index=False)


# --------------------------------------------------
# 3. Split data and train model
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 4. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Lower predicted risk", "Higher predicted risk"],
    zero_division=0,
)

with open(
    RESULTS_DIR / "model_metrics.txt", "w", encoding="utf-8"
) as file:
    file.write("Healthcare Model Explainability - Experiment 8\n")
    file.write("=" * 50 + "\n")
    file.write("Dataset: Synthetic cardiovascular-risk data\n")
    file.write("Model: Random Forest Classifier\n")
    file.write(f"Test accuracy: {accuracy:.4f}\n")
    file.write(f"ROC-AUC: {auc_score:.4f}\n\n")
    file.write(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(
    cm,
    index=["Actual lower risk", "Actual higher risk"],
    columns=["Predicted lower risk", "Predicted higher risk"],
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

predictions = X_test.copy()
predictions["actual_risk"] = y_test
predictions["predicted_risk"] = y_pred
predictions["probability_higher_risk"] = y_prob
predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)


# --------------------------------------------------
# 5. Calculate SHAP values
# --------------------------------------------------

# Explain a manageable sample for global plots
X_explain = X_test.iloc[:100].copy()

explainer = shap.TreeExplainer(model)
raw_values = explainer.shap_values(X_explain)

# Support SHAP versions returning a list per class
# or a three-dimensional array.
if isinstance(raw_values, list):
    shap_values = np.asarray(raw_values[1])
elif np.asarray(raw_values).ndim == 3:
    shap_values = np.asarray(raw_values)[:, :, 1]
else:
    shap_values = np.asarray(raw_values)

expected_value = np.asarray(explainer.expected_value)
if expected_value.ndim == 0:
    base_value = float(expected_value)
else:
    base_value = float(expected_value.reshape(-1)[-1])

# --------------------------------------------------
# 6. Global feature importance
# --------------------------------------------------

mean_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap_value": mean_importance,
}).sort_values("mean_absolute_shap_value", ascending=False)

importance_df.to_csv(
    RESULTS_DIR / "global_feature_importance.csv",
    index=False,
)

# SHAP summary plot
shap.summary_plot(
    shap_values,
    X_explain,
    show=False,
    max_display=len(feature_names),
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_summary.png",
    dpi=250,
    bbox_inches="tight",
)
plt.close()

# SHAP global bar plot
shap.summary_plot(
    shap_values,
    X_explain,
    plot_type="bar",
    show=False,
    max_display=len(feature_names),
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_bar.png",
    dpi=250,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 7. Local explanation for an individual patient
# --------------------------------------------------

patient_index = 0
patient = X_explain.iloc[patient_index]

local_df = pd.DataFrame({
    "feature": feature_names,
    "patient_value": patient.to_numpy(),
    "shap_value": shap_values[patient_index],
})
local_df["absolute_shap_value"] = local_df["shap_value"].abs()
local_df = local_df.sort_values(
    "absolute_shap_value", ascending=False
)

local_df.to_csv(
    RESULTS_DIR / "patient_explanation.csv",
    index=False,
)

# Create a SHAP waterfall plot
explanation = shap.Explanation(
    values=shap_values[patient_index],
    base_values=base_value,
    data=patient.to_numpy(),
    feature_names=feature_names,
)

shap.plots.waterfall(
    explanation,
    max_display=len(feature_names),
    show=False,
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_local_waterfall.png",
    dpi=250,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 8. Save a simple patient explanation report
# --------------------------------------------------

patient_prediction = int(model.predict(patient.to_frame().T)[0])
patient_probability = float(
    model.predict_proba(patient.to_frame().T)[0, 1]
)

with open(
    RESULTS_DIR / "patient_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("Individual Prediction Explanation\n")
    file.write("=" * 35 + "\n")
    file.write("Patient ID: Demonstration sample 1\n")
    file.write(
        "Predicted category: "
        + ("Higher predicted risk" if patient_prediction else
           "Lower predicted risk")
        + "\n"
    )
    file.write(
        f"Model probability for higher-risk class: "
        f"{patient_probability:.4f}\n\n"
    )
    file.write("Features with the largest absolute SHAP values:\n")
    for _, row in local_df.head(5).iterrows():
        file.write(
            f"{row['feature']}: patient value={row['patient_value']:.3f}, "
            f"SHAP value={row['shap_value']:.4f}\n"
        )

# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print("\nExperiment 8 completed successfully!")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc_score:.4f}")
print(f"Patient predicted category: {patient_prediction}")
print(f"All final results saved in: {RESULTS_DIR}")

print("\nGenerated files:")
for path in sorted(RESULTS_DIR.iterdir()):
    print(f" - {path.name}")


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# Create the separate results folder
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="actual")

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open(RESULTS_DIR / "model_metrics.txt", "w",
          encoding="utf-8") as file:
    file.write(f"Model: Random Forest Classifier\n")
    file.write(f"Dataset: Breast Cancer Wisconsin\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write("\n\nClassification Report:\n")
    file.write(classification_report(y_test, y_pred,
                                    target_names=data.target_names))

# Save predictions
predictions = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_pred,
    "actual_label": data.target_names[y_test.to_numpy()],
    "predicted_label": data.target_names[y_pred],
    "probability_class_1": model.predict_proba(X_test)[:, 1],
})
predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

# 5. Calculate SHAP values
explainer = shap.TreeExplainer(model)
X_explain = X_test.iloc[:100].copy()
shap_exp = explainer(X_explain)

# For binary classification, explain class 1 if SHAP
# returns values for multiple classes.
if shap_exp.values.ndim == 3:
    shap_exp = shap_exp[:, :, 1]

# 6. Calculate global feature importance
importance = np.abs(shap_exp.values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_absolute_shap_value": importance,
}).sort_values("mean_absolute_shap_value", ascending=False)

importance_df.to_csv(
    RESULTS_DIR / "global_feature_importance.csv",
    index=False,
)

# 7. Global SHAP summary plot
shap.plots.beeswarm(shap_exp, max_display=15, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_summary.png",
            dpi=300, bbox_inches="tight")
plt.close()

# 8. Global feature importance bar plot
shap.plots.bar(shap_exp, max_display=15, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_bar.png",
            dpi=300, bbox_inches="tight")
plt.close()

# 9. Local explanation for one prediction
shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_waterfall.png",
            dpi=300, bbox_inches="tight")
plt.close()

# 10. Local force plot saved as a static image
shap.plots.force(shap_exp[0], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_local.png",
            dpi=300, bbox_inches="tight")
plt.close()

# 11. Dependence plot for the most important feature
top_feature = importance_df.iloc[0]["feature"]
shap.plots.scatter(shap_exp[:, top_feature], show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_dependence.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Experiment 6 completed successfully!")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Most important feature: {top_feature}")
print(f"All results saved in: {RESULTS_DIR}")
print("\nGenerated files:")
for file_path in sorted(RESULTS_DIR.iterdir()):
    print(f" - {file_path.name}")

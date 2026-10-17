
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Folders and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

CLASS_NAMES = [
    "Normal",
    "Voltage_Sag",
    "Voltage_Swell",
    "Interruption",
    "Harmonics",
    "Flicker",
]

# --------------------------------------------------
# 2. Synthetic PQD signal generation
# --------------------------------------------------

def generate_signal(class_id, n_points=256):
    """Generate a synthetic voltage waveform for a PQD class."""
    t = np.linspace(0, 1, n_points, endpoint=False)
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = 1.0
    signal = np.sin(2 * np.pi * 50 * t + phase)

    if class_id == 1:          # Voltage sag
        amplitude = rng.uniform(0.4, 0.75)
        signal = amplitude * signal

    elif class_id == 2:        # Voltage swell
        amplitude = rng.uniform(1.25, 1.6)
        signal = amplitude * signal

    elif class_id == 3:        # Interruption
        amplitude = rng.uniform(0.02, 0.12)
        signal = amplitude * signal

    elif class_id == 4:        # Harmonics
        signal += (
            0.25 * np.sin(2 * np.pi * 150 * t + phase)
            + 0.12 * np.sin(2 * np.pi * 250 * t)
        )

    elif class_id == 5:        # Flicker / amplitude modulation
        modulation = 1 + 0.25 * np.sin(2 * np.pi * 5 * t)
        signal = modulation * signal

    signal += rng.normal(0, 0.025, n_points)
    return signal


def extract_features(signal):
    """Extract time- and frequency-domain features."""
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1 / (50 * len(signal)))

    fundamental_index = np.argmin(np.abs(frequencies - 50))
    harmonic_mask = frequencies >= 100

    fundamental = spectrum[fundamental_index]
    harmonic_energy = np.sum(spectrum[harmonic_mask] ** 2)
    total_energy = np.sum(spectrum ** 2) + 1e-12

    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "peak": np.max(np.abs(signal)),
        "crest_factor": np.max(np.abs(signal))
        / (np.sqrt(np.mean(signal ** 2)) + 1e-12),
        "skewness": np.mean(
            ((signal - np.mean(signal)) /
             (np.std(signal) + 1e-12)) ** 3
        ),
        "kurtosis": np.mean(
            ((signal - np.mean(signal)) /
             (np.std(signal) + 1e-12)) ** 4
        ),
        "fundamental_magnitude": fundamental,
        "harmonic_energy_ratio": harmonic_energy / total_energy,
        "total_energy": total_energy,
    }


# Create a balanced synthetic dataset
N_PER_CLASS = 250
rows = []
labels = []

for class_id in range(len(CLASS_NAMES)):
    for _ in range(N_PER_CLASS):
        signal = generate_signal(class_id)
        rows.append(extract_features(signal))
        labels.append(class_id)

X = pd.DataFrame(rows)
y = np.array(labels)
feature_names = X.columns.tolist()

# --------------------------------------------------
# 3. Split data and train the classifier
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
y_prob = model.predict_proba(X_test)

# --------------------------------------------------
# 4. Standard performance metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(
    y_test, y_pred, labels=np.arange(len(CLASS_NAMES))
)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    zero_division=0,
)

# Multiclass ROC-AUC
y_test_bin = label_binarize(
    y_test, classes=np.arange(len(CLASS_NAMES))
)
roc_auc = roc_auc_score(
    y_test_bin, y_prob, average="macro", multi_class="ovr"
)

with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write("POWER QUALITY DISTURBANCE CLASSIFICATION\n")
    f.write("=" * 45 + "\n")
    f.write("Dataset: Synthetic PQD signals\n")
    f.write("Model: Random Forest Classifier\n")
    f.write(f"Test samples: {len(y_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report)

pd.DataFrame(
    cm, index=CLASS_NAMES, columns=CLASS_NAMES
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

predictions = pd.DataFrame({
    "actual_class": [CLASS_NAMES[i] for i in y_test],
    "predicted_class": [CLASS_NAMES[i] for i in y_pred],
    "correct": y_test == y_pred,
})
for i, name in enumerate(CLASS_NAMES):
    predictions[f"probability_{name}"] = y_prob[:, i]

predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax)
ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted class",
    ylabel="Actual class",
    title="PQD Confusion Matrix",
)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# Multiclass ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Multiclass ROC Curves",
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
plt.close(fig)

# --------------------------------------------------
# 5. Feature importance and explanation plot
# --------------------------------------------------

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="accuracy",
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv", index=False
)

top_features = importance_df.head(10).sort_values("importance_mean")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features["feature"], top_features["importance_mean"])
ax.set(
    xlabel="Decrease in accuracy after permutation",
    title="Global Feature Importance",
)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
plt.close(fig)

# --------------------------------------------------
# 6. Trustworthiness metrics
# --------------------------------------------------

# Fidelity: agreement between the original model and a
# decision-tree surrogate on held-out test samples.
surrogate = DecisionTreeClassifier(
    max_depth=5, random_state=RANDOM_STATE
)
surrogate.fit(X_train, model.predict(X_train))
surrogate_pred = surrogate.predict(X_test)
fidelity = accuracy_score(y_pred, surrogate_pred)

# Stability: agreement between predictions for nearby
# feature vectors after small relative perturbations.
X_test_np = X_test.to_numpy()
scale = X_train.std(axis=0).to_numpy()
noise = rng.normal(size=X_test_np.shape) * (0.01 * (scale + 1e-8))
X_perturbed = pd.DataFrame(
    X_test_np + noise, columns=feature_names
)
stability = np.mean(
    model.predict(X_test) == model.predict(X_perturbed)
)

# Robustness: prediction agreement under a larger perturbation.
noise_large = rng.normal(size=X_test_np.shape) * (
    0.05 * (scale + 1e-8)
)
X_robust = pd.DataFrame(
    X_test_np + noise_large, columns=feature_names
)
robustness = np.mean(
    model.predict(X_test) == model.predict(X_robust)
)

# Consistency: agreement between two different classifiers.
svm_model = make_pipeline(
    StandardScaler(),
    SVC(probability=True, random_state=RANDOM_STATE),
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
consistency = accuracy_score(y_pred, svm_pred)

# Sparsity: fraction of features required to account for
# 90% of the total permutation importance.
positive_importance = np.maximum(
    importance_df["importance_mean"].to_numpy(), 0
)
if positive_importance.sum() > 0:
    ordered = np.sort(positive_importance)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    required_features = int(np.searchsorted(cumulative, 0.90) + 1)
    sparsity = 1 - required_features / len(feature_names)
else:
    required_features = len(feature_names)
    sparsity = 0.0

trust_metrics = pd.DataFrame([
    {
        "metric": "Fidelity",
        "value": fidelity,
        "interpretation": "Agreement of surrogate and original model",
    },
    {
        "metric": "Stability",
        "value": stability,
        "interpretation": "Prediction agreement under small perturbations",
    },
    {
        "metric": "Consistency",
        "value": consistency,
        "interpretation": "Agreement between Random Forest and SVM",
    },
    {
        "metric": "Robustness",
        "value": robustness,
        "interpretation": "Prediction agreement under larger perturbations",
    },
    {
        "metric": "Sparsity",
        "value": sparsity,
        "interpretation": (
            f"Fraction of features not needed to cover 90% importance; "
            f"{required_features} features cover the target"
        ),
    },
])

trust_metrics.to_csv(
    RESULTS_DIR / "trustworthiness_metrics.csv", index=False
)

# --------------------------------------------------
# 7. Save trustworthiness summary and finish
# --------------------------------------------------

with open(
    RESULTS_DIR / "metrics.txt", "a", encoding="utf-8"
) as f:
    f.write("\n\nTRUSTWORTHINESS METRICS\n")
    f.write("-" * 30 + "\n")
    f.write(trust_metrics.to_string(index=False))
    f.write("\n")

print("\nExperiment 7 completed successfully.")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro ROC-AUC: {roc_auc:.4f}")
print("\nTrustworthiness metrics:")
print(trust_metrics.to_string(index=False))
print(f"\nAll final results are stored in:\n{RESULTS_DIR}")
print("\nGenerated files:")
for path in sorted(RESULTS_DIR.iterdir()):
    print(f" - {path.name}")

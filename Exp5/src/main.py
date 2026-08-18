import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_tabular import LimeTabularExplainer


# Create folders if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------
# STEP 1: CREATE LOAN DATASET
# ---------------------------------------------------

np.random.seed(42)

n_samples = 1000

credit_score = np.random.randint(300, 850, n_samples)
annual_income = np.random.randint(20000, 200000, n_samples)
debt_to_income = np.random.uniform(5, 60, n_samples)
employment_years = np.random.randint(0, 20, n_samples)
loan_amount = np.random.randint(5000, 100000, n_samples)


# Logic for loan approval
loan_approved = (
    (credit_score > 600) &
    (annual_income > 40000) &
    (debt_to_income < 45) &
    (employment_years > 1) &
    (loan_amount < annual_income * 1.5)
).astype(int)


# Create DataFrame
data = pd.DataFrame({
    "Credit_Score": credit_score,
    "Annual_Income": annual_income,
    "Debt_to_Income_Ratio": debt_to_income,
    "Employment_Years": employment_years,
    "Loan_Amount": loan_amount,
    "Loan_Approved": loan_approved
})


# Save dataset
data.to_csv("data/loan_data.csv", index=False)

print("Dataset created successfully!")
print(data.head())


# ---------------------------------------------------
# STEP 2: PREPARE DATA
# ---------------------------------------------------

X = data.drop("Loan_Approved", axis=1)
y = data["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# STEP 3: TRAIN RANDOM FOREST MODEL
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 4: MODEL EVALUATION
# ---------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# STEP 5: SELECT ONE INSTANCE
# ---------------------------------------------------

instance_number = 0

instance = X_test.iloc[instance_number]

print("\nSelected Customer Data:")
print(instance)


# Prediction for selected customer
prediction = model.predict([instance])[0]
prediction_probability = model.predict_proba([instance])[0]

if prediction == 1:
    print("\nModel Prediction: Loan Approved")
else:
    print("\nModel Prediction: Loan Denied")

print("\nPrediction Probabilities:")
print("Denied:", round(prediction_probability[0] * 100, 2), "%")
print("Approved:", round(prediction_probability[1] * 100, 2), "%")


# ---------------------------------------------------
# STEP 6: CREATE LIME EXPLAINER
# ---------------------------------------------------

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Denied", "Approved"],
    mode="classification",
    random_state=42
)


# ---------------------------------------------------
# STEP 7: GENERATE LIME EXPLANATION
# ---------------------------------------------------

explanation = explainer.explain_instance(
    data_row=instance.values,
    predict_fn=model.predict_proba,
    num_features=5
)


# Print explanation
print("\nLIME Explanation:")

for feature, importance in explanation.as_list():
    print(feature, ":", round(importance, 4))


# ---------------------------------------------------
# STEP 8: SAVE INTERACTIVE HTML EXPLANATION
# ---------------------------------------------------

explanation.save_to_file("outputs/lime_explanation.html")

print("\nLIME HTML explanation saved successfully!")


# ---------------------------------------------------
# STEP 9: VISUALIZE LIME FEATURE IMPORTANCE
# ---------------------------------------------------

features = explanation.as_list()

feature_names = [item[0] for item in features]
importance_values = [item[1] for item in features]


plt.figure(figsize=(10, 6))

plt.barh(feature_names, importance_values)

plt.xlabel("Feature Contribution")
plt.ylabel("Features")
plt.title("LIME Feature Importance for Selected Customer")

plt.axvline(x=0, linewidth=1)

plt.tight_layout()

plt.savefig(
    "outputs/lime_feature_importance.png",
    dpi=300
)

plt.show()


print("\nExperiment completed successfully!")
print("Check the outputs folder for:")
print("1. lime_explanation.html")
print("2. lime_feature_importance.png")
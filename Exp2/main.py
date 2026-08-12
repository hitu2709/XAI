import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Create Dataset
# --------------------------------------------------

data = pd.DataFrame({
    "Size_sqft": [1400, 1600, 1700, 1875, 1100,
                  1550, 1230, 1900, 2000, 1750],

    "Bedrooms": [3, 3, 3, 4, 2,
                 3, 2, 4, 4, 3],

    "Location_Score": [8, 7, 9, 6, 5,
                       7, 6, 9, 8, 7],

    "Price": [245000, 312000, 279000, 308000, 199000,
              265000, 180000, 330000, 355000, 290000]
})

print("Dataset:")
print(data)


# --------------------------------------------------
# 2. Visualize Dataset
# --------------------------------------------------

sns.pairplot(data)
plt.show()


# --------------------------------------------------
# 3. Prepare Data
# --------------------------------------------------

X = data[["Size_sqft", "Bedrooms", "Location_Score"]]

y = data["Price"]


# --------------------------------------------------
# 4. Split Data into Training and Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. Create and Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Get Intercept and Coefficients
# --------------------------------------------------

intercept = model.intercept_

coefficients = model.coef_

print("\n==============================")
print("MODEL COEFFICIENTS")
print("==============================")

print("Intercept:", intercept)

for feature, coefficient in zip(X.columns, coefficients):

    print(feature, ":", coefficient)


# --------------------------------------------------
# 7. Interpret Coefficients
# --------------------------------------------------

print("\n==============================")
print("COEFFICIENT INTERPRETATION")
print("==============================")

for feature, coefficient in zip(X.columns, coefficients):

    if coefficient > 0:

        print(
            f"For every 1 unit increase in {feature}, "
            f"the house price increases by "
            f"{coefficient:.2f}, keeping other factors constant."
        )

    else:

        print(
            f"For every 1 unit increase in {feature}, "
            f"the house price decreases by "
            f"{abs(coefficient):.2f}, keeping other factors constant."
        )


# --------------------------------------------------
# 8. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


print("\n==============================")
print("ACTUAL VS PREDICTED")
print("==============================")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(comparison)


# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Mean Squared Error (MSE):", mse)

print("R-squared Score:", r2)


# --------------------------------------------------
# 10. Actual vs Predicted Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    alpha=0.7
)

plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="dashed"
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.show()
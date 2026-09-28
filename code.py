import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("electricity_consumption.csv")

features = [
    "Temperature_C", "Humidity_Percent", "Household_Size",
    "Appliance_Hours", "Previous_Day_kWh"
]

X = data[features]
y = data["Daily_Consumption_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree Regression":
        DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest Regression":
        RandomForestRegressor(n_estimators=100, max_depth=6,
                              random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel:", name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# Gradient Descent for Linear Regression
# Single feature: Previous_Day_kWh

X_gd = data[["Previous_Day_kWh"]].values
y_gd = data["Daily_Consumption_kWh"].values

Xtr, Xte, ytr, yte = train_test_split(
    X_gd, y_gd, test_size=0.20, random_state=42
)

mean = Xtr.mean(axis=0)
std = Xtr.std(axis=0)

Xtr = (Xtr - mean) / std
Xte = (Xte - mean) / std

w = 0.0
b = 0.0
learning_rate = 0.03
epochs = 1000

for i in range(epochs):
    y_pred = w * Xtr[:, 0] + b
    error = y_pred - ytr

    dw = (2 / len(Xtr)) * np.sum(Xtr[:, 0] * error)
    db = (2 / len(Xtr)) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

y_pred = w * Xte[:, 0] + b

print("\nGradient Descent")
print("Weight:", w)
print("Bias:", b)
print("MAE:", mean_absolute_error(yte, y_pred))
print("MSE:", mean_squared_error(yte, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(yte, y_pred)))
print("R2 Score:", r2_score(yte, y_pred))

 # comparison to regression model as baseline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load and preprocess dataset
df = pd.read_csv("WEC_Perth_49.csv")
df = df.drop(columns=['qW'])

X = df[[col for col in df.columns if col.startswith('X') or col.startswith('Y')]]
y = df['Total_Power']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = lr.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
percent_error = (mae / y_test.mean()) * 100  # Mean Percent Error

print(f"Linear Regression MAE: {mae:.2f}")
print(f"Linear Regression MSE: {mse:.2f}")
print(f"Linear Regression Percent Error: {percent_error:.2f}%")

    # %%
import matplotlib.pyplot as plt

# Plot actual vs predicted values for linear regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='darkorange', label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Ideal Fit (y = x)')
plt.xlabel("Actual Total Power (W)")
plt.ylabel("Predicted Total Power (W)")
plt.title("Linear Regression: Predicted vs Actual Total Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

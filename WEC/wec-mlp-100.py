import pandas as pd

# Load the Perth 49-buoy dataset
df = pd.read_csv("WEC_Perth_100.csv")
# %%
# Show basic info
print(df.shape)
print(df.columns[:10])  # Show first 10 column names
print(df[['Total_Power', 'qW']].head())  # Show target columns

# %%
# Drop the qW column (not needed for our model)
df = df.drop(columns=['qW'])

# Separate features (X1 to Y49) and target (Total_Power)
# Keep only columns that start with 'X' or 'Y' (positions only)
X = df[[col for col in df.columns if col.startswith('X') or col.startswith('Y')]]
y = df['Total_Power']

# Confirm dimensions
print("Input shape:", X.shape)
print("Target shape:", y.shape)

# %%
from sklearn.model_selection import train_test_split
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# %%
from sklearn.preprocessing import StandardScaler

# Create and fit the scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First row (scaled):")
print(X_train_scaled[0])

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# %%
# Improved model with smaller learning rate and more iterations
model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0005,   # smaller step size
    max_iter=1000,               # more training time
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Improved MSE: {mse:.2f}")
print(f"Improved MAE: {mae:.2f}")

# %%
# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='seagreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Total Power (W)")
plt.ylabel("Predicted Total Power (W)")
plt.title("Improved Model: Predicted vs Actual Total Power")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
percent_error = (mae / y_test.mean()) * 100
print(f"Mean Percent Error: {percent_error:.2f}%")
# %%
# Testing the model on a different dataset
# Load Sydney dataset
sydney_df = pd.read_csv("WEC_Sydney_100.csv")

# Drop qW and power columns — keep only X/Y positions
sydney_df = sydney_df.drop(columns=['qW'])
sydney_X = sydney_df[[col for col in sydney_df.columns if col.startswith('X') or col.startswith('Y')]]
sydney_y = sydney_df['Total_Power']

sydney_X_scaled = scaler.transform(sydney_X)  # DO NOT fit again!

sydney_pred = model.predict(sydney_X_scaled)

# Evaluate performance
sydney_mse = mean_squared_error(sydney_y, sydney_pred)
sydney_mae = mean_absolute_error(sydney_y, sydney_pred)
sydney_percent_error = (sydney_mae / sydney_y.mean()) * 100

print(f"Test on Sydney Data — MSE: {sydney_mse:.2f}")
print(f"MAE: {sydney_mae:.2f}")
print(f"Percent Error: {sydney_percent_error:.2f}%")

# %%
# Plot actual vs predicted for Sydney dataset
plt.figure(figsize=(10, 5))
plt.scatter(sydney_y, sydney_pred, alpha=0.4, color='darkorange')
plt.plot([sydney_y.min(), sydney_y.max()], [sydney_y.min(), sydney_y.max()], 'r--', linewidth=2)
plt.xlabel("Actual Total Power (W) - Sydney")
plt.ylabel("Predicted Total Power (W) by Perth-trained Model")
plt.title("Generalization Test: Perth Model on Sydney Data")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
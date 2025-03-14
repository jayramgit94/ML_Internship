import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("sock Prices Data Set.csv")

# Step 2: Display dataset overview
print("\n📌 Dataset Overview:")
print(df.head())  # Show first 5 rows
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())  # Check for NaN values

# Step 3: Handle missing values by filling with column mean
df[['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(df[['open', 'high', 'low']].mean())

# Step 4: Check column names and select relevant features
print("\n🔍 Column Names in the Dataset:", df.columns)

# Step 5: Define Features (X) and Target Variable (y)
X = df[['open', 'high', 'low']]  # Features (Stock prices at different times)
y = df['close']  # Target (Stock closing price)

# Step 6: Split data into training (80%) & testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 10: Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R-squared Score (R²): {r2:.2f}")

# Step 11: Visualize Predictions vs Actual Prices
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(y_pred, label="Predicted Price", color="red", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Stock Price Prediction: Actual vs Predicted")
plt.show()

# Step 12: Save Processed Dataset
df.to_csv("processed_stock_prices.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_stock_prices.csv'")

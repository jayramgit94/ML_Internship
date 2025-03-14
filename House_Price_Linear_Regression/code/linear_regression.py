import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define column names manually (since dataset has no headers)
column_names = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5",
                "Feature6", "Feature7", "Feature8", "Feature9", "Feature10",
                "Feature11", "Feature12", "Feature13", "price"]  # Last column is target variable

# Load dataset with proper delimiter
df = pd.read_csv("house Prediction Data Set.csv", delim_whitespace=True, names=column_names)

# Display first few rows to check correct loading
print("\nDataset Overview:")
print(df.head())

# Save the processed dataset
df.to_csv("processed_house.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_house.csv'")



# Step 1: Check for missing values
print("\nMissing Values Before Processing:")
print(df.isnull().sum())

# Fill missing values (if any) with column mean
df.fillna(df.mean(), inplace=True)

# Step 2: Define features (X) and target variable (y)
X = df.drop(columns=['price'])  # Features
y = df['price']  # Target variable (house price)

# Step 3: Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on test data
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 7: Plot Actual vs. Predicted House Prices
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()

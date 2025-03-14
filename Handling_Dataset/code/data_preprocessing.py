import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_name = "1) iris.csv"  # Change filename if needed
df = pd.read_csv(file_name)

# Display dataset info
print("\nDataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 1: Handling Missing Values (Only for numeric columns)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Step 2: Encoding Categorical Variables
categorical_columns = df.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print("\nEncoding Categorical Variables:", categorical_columns.tolist())
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

# Step 3: Feature Scaling (Standardization)
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nStandardized Data Sample:")
print(df.head())

# Step 4: Splitting Data into Training & Testing Sets
target_column = df.columns[-1]  # Assuming last column is target
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 5: Save Processed Data
processed_file_name = f"processed_{file_name}"
df.to_csv(processed_file_name, index=False)
print(f"\n✅ Data preprocessing completed! Processed file saved as: {processed_file_name}")

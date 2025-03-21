import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv("churn-bigml-80.csv")  # Use the bigger dataset for better training

# Display basic dataset information
print("\n📌 Dataset Overview:")
print(df.head())

# Step 2: Handle missing values
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())  # Show missing values per column

df.dropna(inplace=True)  # Drop rows with missing values
df.to_csv("processed_churn.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_churn.csv'")



# import pandas as pd

df = pd.read_csv("processed_churn.csv")
print(df.head())  # Display first 5 rows




# Step 3: Convert categorical values to numeric
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns  # Find categorical columns

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode each categorical column
    label_encoders[col] = le

# Step 4: Define Features (X) and Target Variable (y)
X = df.drop(columns=['Churn'])  # Independent variables
y = df['Churn']  # Target variable (0 = No Churn, 1 = Churn)

# Step 5: Standardize Numerical Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split Data into Training (80%) & Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Confusion Matrix Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 11: Save Processed Dataset
df.to_csv("processed_churn.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_churn.csv'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")

# Step 2: Display dataset overview
print("\n📌 Dataset Overview:")
print(df.head())

# Step 3: Convert categorical values to numeric (if needed)
if df.dtypes['species'] == object:  
    df['species'] = df['species'].astype('category').cat.codes  # Encode species labels

# Step 4: Define Features (X) and Target Variable (y)
X = df.drop(columns=['species'])  # Independent variables
y = df['species']  # Target variable

# Step 5: Split Data into Training (80%) & Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Decision Tree Model
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Visualize Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Step 10: Save Processed Dataset
df.to_csv("processed_iris_decision_tree.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_iris_decision_tree.csv'")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("Sentiment dataset.csv")  
print("\nColumn Names in the Dataset:")
print(df.columns)  # Display actual column names

# Step 2: Handle incorrect column names
df.rename(columns={"Text": "text", "Sentiment": "sentiment"}, inplace=True)

# Step 3: Display dataset overview
print("\nDataset Overview:")
print(df.head())

# Step 4: Remove missing sentiment values
df = df.dropna(subset=['sentiment'])  # Fix NaN issue
df['sentiment'] = df['sentiment'].map({"Positive": 1, "Negative": 0, "Neutral": 2})  

# Step 5: Check if `y` still contains NaN values
print("\nMissing Values in Sentiment Column After Processing:")
print(df['sentiment'].isnull().sum())

df = df.dropna(subset=['sentiment'])  # Remove rows with NaN in sentiment
df = df.dropna(subset=['sentiment'])  # Drop all rows with NaN sentiment
df['sentiment'] = df['sentiment'].map({"Positive": 1, "Negative": 0, "Neutral": 2})  # Convert labels
df = df.dropna(subset=['sentiment'])  # Drop any remaining NaN values in sentiment
df['sentiment'] = df['sentiment'].astype(int)  # Convert to integer


# Step 7: Save the processed dataset
df.to_csv("processed_sentiment.csv", index=False)
print("\n✅ Processed dataset saved as 'processed_sentiment.csv'")

# Step 8: Define features (X) and target variable (y)
X = df['text'].dropna()  # Drop missing text values
X = X[X.str.strip() != ""]  # Remove empty strings


# Ensure there are no empty strings or missing values
X = X[X.str.strip() != ""]  # Remove empty rows

y = df['sentiment']

# Step 9: Convert text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Step 10: Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 11: Train a K-Nearest Neighbors (KNN) classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 12: Make predictions on the test set
y_pred = model.predict(X_test)


print("\n🔍 Checking if text column is empty:")
print(X.head())  # Show first 5 values of X
print(f"Total non-empty text entries: {X.notnull().sum()}")


# Step 13: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 14: Test with a new review
new_review = ["This product is amazing! I love it."]
new_review_tfidf = vectorizer.transform(new_review)

predicted_sentiment = model.predict(new_review_tfidf)
sentiment_label = {1: "Positive", 0: "Negative", 2: "Neutral"}
print(f"\nPredicted Sentiment: {sentiment_label[predicted_sentiment[0]]}")


import matplotlib.pyplot as plt
import seaborn as sns

# Step 14: Plot Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["sentiment"], palette="coolwarm")
plt.xticks(ticks=[0, 1, 2], labels=["Negative", "Positive", "Neutral"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution")
plt.show()

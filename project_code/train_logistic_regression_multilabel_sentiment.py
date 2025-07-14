import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (classification_report,accuracy_score,f1_score,confusion_matrix)
import joblib

# Load dataset
df = pd.read_csv(r'datasets\labeled_data\ethiopian_airlines_overall_and_topic_sentiment.csv')

# Prepare input features and multilabel targets
X_text = df["review_comment"].fillna("")
y_labels = df[[col for col in df.columns if col.endswith("_sentiment") and col != "overall_sentiment"]]
y_numeric = y_labels.apply(lambda col: col.map({"Negative": 0, "Neutral": 1, "Positive": 2}).fillna(1))

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Train model
model = MultiOutputClassifier(LogisticRegression(max_iter=5000,class_weight="balanced"))
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Labels for reporting
labels = ["Negative", "Neutral", "Positive"]

# Per-category metrics and confusion matrix
for idx, col in enumerate(y_labels.columns):
    print(f"\n=== Logistic Regression Category: {col} ===")

    # Print classification report
    report = classification_report(y_test.iloc[:, idx], y_pred[:, idx], target_names=labels, output_dict=True, zero_division=0)

    print(classification_report( y_test.iloc[:, idx], y_pred[:, idx], target_names=labels, zero_division=0))

    # show accuracy for that category
    acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
    print(f"Accuracy for {col}: {acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test.iloc[:, idx], y_pred[:, idx])

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {col}")
    plt.tight_layout()
    plt.show()

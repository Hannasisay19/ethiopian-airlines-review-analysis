import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

# Train model with balanced class weights
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42))
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Labels for reporting
labels = ["Negative", "Neutral", "Positive"]

# Per-category metrics and confusion matrix
for idx, col in enumerate(y_labels.columns):
    print(f"\n=== Random Forest Category: {col} ===")

    # Print classification report
    report = classification_report(y_test.iloc[:, idx], y_pred[:, idx], target_names=labels, zero_division=0)
    print(report)

    # Accuracy
    acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
    print(f"Accuracy for {col}: {acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test.iloc[:, idx], y_pred[:, idx])

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,xticklabels=labels,yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {col}")
    plt.tight_layout()
    plt.show()

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
f1_scores = []

for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y_numeric.iloc[train_idx], y_numeric.iloc[val_idx]

    fold_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42))
    fold_model.fit(X_train_fold, y_train_fold)
    y_val_pred = fold_model.predict(X_val_fold)

    acc = []
    f1 = []
    for i in range(y_numeric.shape[1]):
        acc.append(accuracy_score(y_val_fold.iloc[:, i], y_val_pred[:, i]))
        f1.append(f1_score(y_val_fold.iloc[:, i], y_val_pred[:, i], average='macro'))

    accuracies.append(np.mean(acc))
    f1_scores.append(np.mean(f1))

print(f"\nRandom Forest Avg Cross-Validated Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"Random Forest Avg Cross-Validated Macro F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

# Save trained model and vectorizer
joblib.dump(model, 'models/random_forest_topic_sentiment_classifier.joblib')
joblib.dump(vectorizer, 'models/tfidf_vectorizer_topic_sentiment_by_rf.joblib')
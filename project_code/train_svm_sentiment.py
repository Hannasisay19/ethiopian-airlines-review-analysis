import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import os


# Load data
df = pd.read_csv(r'datasets\sentiment_analysis\ethiopian_airlines_overall_sentiment.csv')

# Extract input and target
X = df['review_comment']
y = df['overall_sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode labels 
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train_encoded)

# Train SVM (with calibration for probability support)
svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm_model = CalibratedClassifierCV(svm_base, method='sigmoid')
svm_model.fit(X_resampled, y_resampled)

# predictions
y_pred_encoded = svm_model.predict(X_test_vec)
y_pred_labels = le.inverse_transform(y_pred_encoded)

# Evaluation
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_labels))
print(f"Accuracy: {accuracy_score(y_test, y_pred_labels):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM on Ethiopian Airlines Reviews')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.show()

# Save model, vectorizer, and encoder
joblib.dump(svm_model, "models/final_svm_model.joblib")

print("\n Model saved in 'models/' directory.")


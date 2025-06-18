from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv(r'datasets\sentiment_analysis\ethiopian_airlines_overall_sentiment.csv')
X = df['review_comment']
y = df['overall_sentiment']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode labels (keeps original labels for reporting)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Vectorize training data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train_encoded)

# Train calibrated SVM model on resampled data
svm_model = CalibratedClassifierCV(estimator=SVC(kernel='rbf', class_weight='balanced', probability=True),method='sigmoid')

# Train
svm_model.fit(X_resampled, y_resampled)

# Vectorize test data
X_test_vec = vectorizer.transform(X_test)

# Make predictions
y_pred_encoded = svm_model.predict(X_test_vec)
y_pred_labels = le.inverse_transform(y_pred_encoded)


print("SVM Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,y_pred_labels),
           annot=True, fmt='d',
           cmap='Reds',
           xticklabels=le.classes_,
           yticklabels=le.classes_)
plt.title('SVM Confusion Matrix')
plt.show()



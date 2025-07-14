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

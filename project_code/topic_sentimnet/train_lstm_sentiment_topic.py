import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset

file_path = r"datasets\labeled_data\ethiopian_airlines_overall_and_topic_sentiment.csv"
df = pd.read_csv(file_path)

# Prepare labels
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
labels = ["Negative", "Neutral", "Positive"]

y_labels = df[[col for col in df.columns if col.endswith("_sentiment") and col != "overall_sentiment"]]
y_numeric = y_labels.apply(lambda col: col.map(label_map).fillna(1))

# Vectorize text
X_text = df["review_comment"].fillna("")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X_text).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_numeric.values, test_size=0.2, random_state=42)

# Custom Dataset
class TfidfDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TfidfDataset(X_train, y_train)
test_dataset = TfidfDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_labels):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels * output_dim)
        self.output_dim = output_dim
        self.num_labels = num_labels

    def forward(self, x):
        # Reshape for LSTM input: (batch, seq_len=1, input_dim)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        logits = self.fc(out)
        # Split logits per label
        logits = logits.view(-1, self.num_labels, self.output_dim)
        return logits

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = 3   # 3 classes
num_labels = y_train.shape[1]

model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_labels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = 0
        for i in range(num_labels):
            loss += criterion(outputs[:, i, :], y_batch[:, i])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=2).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Per-category metrics
for i, col in enumerate(y_labels.columns):
    print(f"\n=== LSTM Category: {col} ===")
    print(classification_report(all_targets[:, i], all_preds[:, i], target_names=labels, zero_division=0))
    acc = accuracy_score(all_targets[:, i], all_preds[:, i])
    print(f"Accuracy for {col}: {acc:.3f}")

    cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {col}")
    plt.tight_layout()
    plt.show()

# Save model and vectorizer
#torch.save(model.state_dict(), "LSTM/lstm_sentiment_multilabel_model.pt")
#joblib.dump(vectorizer, "LSTM/tfidf_vectorizer_sentiment.joblib")

torch.save(model.state_dict(), r"models\LSTM_topic\lstm_sentiment_multilabel_model.pt")
joblib.dump(vectorizer, r"models\LSTM_topic\lstm_tfidf_vectorizer_sentiment.joblib")


print("\nLSTM model and vectorizer saved successfully.")

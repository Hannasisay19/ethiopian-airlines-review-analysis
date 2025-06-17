import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Data Preparation
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. LSTM Model Architecture
class SentimentLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim, n_layers, dropout, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim,
                            num_layers=n_layers, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = outputs.last_hidden_state
        lengths = attention_mask.sum(dim=1)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

# 3. Training Function
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    class_correct = [0] * model.fc.out_features
    class_total = [0] * model.fc.out_features
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
    
        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

    per_class_accuracy = [c / t if t != 0 else 0 for c, t in zip(class_correct, class_total)]
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), per_class_accuracy

# 4. Evaluation Function
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (total_loss / len(data_loader), 
            correct_predictions.double() / len(data_loader.dataset),
            all_preds, all_labels)

# 5. Main Execution
def main():
    df = pd.read_csv(r'datasets\sentiment_analysis\ethiopian_airlines_overall_sentiment.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_comment'], df['overall_sentiment'], test_size=0.2, random_state=42, stratify=df['overall_sentiment']
    )
    
    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Tokenizer (using HuggingFace's tokenizer for better text handling)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    # Create datasets
    max_len = 200
    batch_size = 128
    
    train_dataset = TextDataset(X_train.tolist(), y_train_encoded, tokenizer, max_len)
    test_dataset = TextDataset(X_test.tolist(), y_test_encoded, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SentimentLSTM(
        bert_model_name='bert-base-uncased',
        hidden_dim=64,
        output_dim=num_classes,
        n_layers=2,
        dropout=0.3
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  
    
    # Training loop
    num_epochs = 15
    best_accuracy = 0
    patience, counter = 3, 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):  
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_model(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'models/final_lstm_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Final evaluation
    model.load_state_dict(torch.load(r'models/final_lstm_model.pth'))
    _, _, y_pred_encoded, y_true = eval_model(model, test_loader, criterion, device)
    y_pred = le.inverse_transform(y_pred_encoded)
    y_test_labels = le.inverse_transform(y_true)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
import os
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# Define paths
csv_path = r"C:\Users\Acer\Desktop\sentiment_analysis\sentiment140_data\training.1600000.processed.noemoticon.csv"

# Check if file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=[0, 5], names=["sentiment", "text"], on_bad_lines="skip")

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower().strip()  # Convert to lowercase & remove extra spaces

df["text"] = df["text"].astype(str).apply(preprocess_text)

# Convert sentiment labels: 4 -> Positive, 0 -> Negative, Others -> Neutral (2)
df["sentiment"] = df["sentiment"].map({4: 4, 0: 0}).fillna(2).astype(int)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = encode_texts(X_train.tolist())
test_encodings = encode_texts(X_test.tolist())

# Convert labels to tensors
train_labels = torch.tensor(y_train.tolist())
test_labels = torch.tensor(y_test.tolist())

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)

# DataLoader (Batch Processing)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Pre-trained BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training Loop
num_epochs = 3
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Step
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

# Model Evaluation
model.eval()
predictions, true_labels = [], []

for batch in test_loader:
    input_ids, attention_mask, labels = [b.to(device) for b in batch]
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Print Metrics
print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))
print("Classification Report:\n", classification_report(true_labels, predictions))

# Sentiment Prediction Function
def predict_sentiment(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    encodings = encode_texts(processed_texts)
    
    input_ids, attention_mask = encodings["input_ids"].to(device), encodings["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    
    sentiment_map = {4: "Positive", 0: "Negative", 2: "Neutral"}
    return [sentiment_map[pred] for pred in predictions]

# Example Sentiment Predictions
example_texts = ["I love this movie! It's amazing.", "I hate this product, it's terrible.", "It's an okay product."]
predicted_sentiments = predict_sentiment(example_texts)

for text, sentiment in zip(example_texts, predicted_sentiments):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n---")

# Save Model & Tokenizer
model_dir = "bert_sentiment_model"
tokenizer_dir = "bert_sentiment_tokenizer"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(tokenizer_dir)

print(f"Model & Tokenizer saved to {model_dir} and {tokenizer_dir}")

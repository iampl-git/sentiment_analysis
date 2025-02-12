import zipfile
import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
ZIP_PATH = "C:/Users/Acer/Desktop/sentiment_analysis/training.1600000.processed.noemoticon.csv.zip"
EXTRACT_FOLDER = "sentiment140_data"
CSV_FILE = os.path.join(EXTRACT_FOLDER, "training.1600000.processed.noemoticon.csv")
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Extract data
if not os.path.exists(EXTRACT_FOLDER):
    os.makedirs(EXTRACT_FOLDER)

if not os.path.exists(CSV_FILE):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

# Load data
df = pd.read_csv(CSV_FILE, encoding="ISO-8859-1", usecols=[0, 5], names=["sentiment", "text"])

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()
    return text

df['text'] = df['text'].apply(preprocess_text)

# Train-test split
X = df['text']
y = df['sentiment'].map({0: 0, 4: 1})  # Convert 0 to Negative, 4 to Positive
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model trained and saved successfully!")

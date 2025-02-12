import zipfile
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define paths to the zip file and the extraction folder
zip_path = "C:/Users/Acer/Desktop/sentiment_analysis/training.1600000.processed.noemoticon.csv.zip"
extract_folder = "sentiment140_data"

# Step 1: Extract the dataset from the zip file if not already extracted
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Get the path to the CSV file after extraction
csv_path = os.path.join(extract_folder, "training.1600000.processed.noemoticon.csv")

# Step 2: Load the dataset into a DataFrame
print("Loading dataset...")
df = pd.read_csv(csv_path, encoding="ISO-8859-1", usecols=[0, 5], names=["sentiment", "text"])

# Step 3: Preprocess the text data (basic cleaning)
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (usernames)
    text = re.sub(r'@\S+', '', text)
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

df['text'] = df['text'].apply(preprocess_text)

# Step 4: Split the data into training and testing sets (80% train, 20% test)
X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_tfidf)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 8: Make predictions on new text data
new_text = ["I love this movie! It's amazing.", "I hate this product, it's terrible."]
new_text_processed = [preprocess_text(text) for text in new_text]
new_text_tfidf = vectorizer.transform(new_text_processed)
predictions = model.predict(new_text_tfidf)

# Print predictions for new text data
for text, sentiment in zip(new_text, predictions):
    sentiment_label = "Positive" if sentiment == 4 else "Negative"  # Sentiment value 4 corresponds to positive, 0 to negative
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print("---")

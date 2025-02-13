from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer once at the start
try:
    # Replace with a popular sentiment analysis model or a local model path
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Define sentiment labels
def sentiment_label(sentiment_value):
    labels = {
        4: "Very Positive",  # Highest sentiment score
        3: "Positive",       # Slightly positive sentiment
        2: "Neutral",        # Neutral sentiment
        1: "Negative",       # Slightly negative sentiment
        0: "Very Negative"   # Lowest sentiment score
    }
    return labels.get(sentiment_value, "Unknown")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    if not text:
        return render_template('index.html', prediction="Please enter some text.")
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        
    sentiment = sentiment_label(prediction)
    
    # Debugging: print the prediction and sentiment to console
    print(f"Prediction: {prediction} -> Sentiment: {sentiment}")
    
    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__': 
    app.run(debug=True)  # Remember to set debug=False in production!
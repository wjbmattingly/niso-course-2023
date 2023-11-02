import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Function to classify the text
# Function to classify the text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

tokenizer, model = load_model()

# Define default sentences
default_sentences = [
    "Hello, my dog is cute",
    "I had a wonderful day at the park",
    "I'm feeling a bit under the weather today",
    "I hated the movie.",
    "I did not dislike the movie.",
    "There is no way I am ever going to see that movie again.",
    "There is no way I am never going to see that movie again.",
    "There is no way I am not never going to see that movie again.",
    "The film was filmed in France",
    "The film was filmed in Afghanistan",
    "Custom Sentence..."
]

# Streamlit interface
st.title('Sentiment Analysis with DistilBERT')

# Dropdown for default sentences or custom input
sentence_selection = st.selectbox('Choose a sentence or write your own:', default_sentences)

# Check if the user selected custom sentence
if sentence_selection == "Custom Sentence...":
    sentence = st.text_area("Write your sentence here:")
else:
    sentence = sentence_selection

# Button to classify text
if st.button('Classify'):
    if sentence:
        # Classify the sentence
        logits = classify_text(sentence)
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=1)
        # Get the highest probability class
        predicted_class_id = probabilities.argmax().item()
        # Convert the predicted class id to label
        predicted_label = "Positive" if predicted_class_id == 1 else "Negative"
        # Display the result
        st.write(f'Predicted Sentiment: **{predicted_label}**')
        st.write(f'Confidence: **{probabilities[0][predicted_class_id].item():.4f}**')
    else:
        st.warning("Please input a sentence or select a default sentence.")
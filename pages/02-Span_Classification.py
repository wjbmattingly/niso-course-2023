import streamlit as st
import spacy
from spacy import displacy

# Function to download and load the spaCy model
@st.cache_resource
def load_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# Load the spaCy model
nlp = load_model()

# Default paragraphs with various entities
default_texts = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "San Francisco considers banning sidewalk delivery robots",
    "London is a very big city in the United Kingdom.",
    "Elon Musk has shared his vision of colonizing Mars.",
    ("In Johannesburg, South Africa’s biggest city, Thabo Mbeki spoke at the convention."),
    ("Graça Machel, a humanitarian from Mozambique, was married to Nelson Mandela, "
     "the former president of South Africa."),
    "Custom Text - write your own."
]

# Streamlit interface
st.title('Named Entity Recognition with spaCy')

# Dropdown to select default text or custom text
selected_text = st.selectbox("Choose a text for NER analysis or write your own:", default_texts)

# Text area for custom text input
text_to_analyze = ""
if selected_text == "Custom Text - write your own.":
    custom_text = st.text_area("Enter your text:", height=150)
    # Submit button for custom text
    submit_button = st.button(label="Analyze Text")
    if submit_button:
        text_to_analyze = custom_text
else:
    text_to_analyze = selected_text

if text_to_analyze:
    # Perform NER
    doc = nlp(text_to_analyze)
    # Display NER using spaCy's displaCy
    html = displacy.render(doc, style="ent")
    st.markdown(html, unsafe_allow_html=True)
else:
    # Only show the warning if the submit button was clicked without any custom text
    if 'submit_button' in locals() and submit_button:
        st.warning("Please enter some text to analyze.")

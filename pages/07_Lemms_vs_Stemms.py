import streamlit as st
import spacy
import nltk

nltk.download("wordnet")

# Initialize NLTK stemmer and lemmatizer
nltk_stemmer = nltk.stem.PorterStemmer()
nltk_lemmatizer = nltk.stem.WordNetLemmatizer()

# Load spaCy English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample texts for processing
sample_texts = [
    "The children are running rapidly and the games have started.",
    "She had a better understanding after reading the book.",
    "The cats are sitting on the windowsill.",
    "They pursued their studies every day."
]

# Streamlit application
def main():
    # App title
    st.title('Stemming vs. Lemmatization')

    # Display sample texts in a selectbox
    text = st.selectbox('Choose a sample text:', sample_texts)

    # NLTK Stemming
    nltk_stemmed_words = [nltk_stemmer.stem(word) for word in text.split()]


    # NLTK Lemmatization
    nltk_lemmatized_words = [nltk_lemmatizer.lemmatize(word) for word in text.split()]


    # spaCy Lemmatization
    spacy_doc = nlp(text)
    spacy_lemmatized_words = [token.lemma_ for token in spacy_doc]

    # Display results side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**NLTK Stemming**")
        st.write(' '.join(nltk_stemmed_words))

    with col2:
        st.markdown("**NLTK Lemmatization**")
        st.write(' '.join(nltk_lemmatized_words))

    with col3:
        st.markdown("**spaCy Lemmatization**")
        st.write(' '.join(spacy_lemmatized_words))

if __name__ == '__main__':
    main()

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
    "They pursued their studies every day.",
    "Write Custom Sentence"  # Add this option for custom text input
]

# Streamlit application
def main():
    # App title
    st.title('Stemming vs. Lemmatization')

    mode = st.selectbox("Select Mode", ["Experiment", "Exercise"])

    if mode == "Experiment":
        # Display sample texts in a selectbox
        selected_option = st.selectbox('Choose a sample text or write your own:', sample_texts)

        # Check if user selected the custom text option
        if selected_option == "Write Custom Sentence":
            text = st.text_input("Write your sentence here:")  # Display text input
        else:
            text = selected_option

            # Only process if text is available
            if text:
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
    else:
        st.write("Your goal is to write a query that returns a hit for 'friends running' in each of the lines below.")
        texts = [
            "The friends ran to the store.",
            "The friends run to to the store.",
            "The friend runs fast.",
            "The friend ran away.",
            "The friend is running."
        ]
        st.write(texts)
        
        query = st.text_input("Write your query here")
        option = st.checkbox("Use lemmas?")
        col1, col2 = st.columns(2)
        if st.button("Test query"):
            for text in texts:
                col1.write(text)
                lemma = " ".join(token.lemma_ for token in nlp(text))
            
                if option:
                    query_text = lemma
                else:
                    query_text = text
                if query in query_text:
                    col2.write("Good job! This matched!")
                else:
                    col2.write("This did not match.")

if __name__ == '__main__':
    main()

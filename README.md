# Bio

William J.B. Mattingly, PhD is a Postdoctoral Fellow in the Smithsonian Institution's Data Science Lab. He specializes in the application of machine learning and natural language processing on archival and historical documents. He is the author of Introduction to Python for Digital Humanists (2023), content creator for Python Tutorials for Digital Humanities, and lead developer for the Bitter Aloe Project.

# Course Summary

This comprehensive course is designed to equip students with the essential skills and knowledge required to undertake text and data mining tasks. Throughout this course, students will be introduced to key concepts and tools of text and data mining, including data types, data structures, data pre-processing, text processing, data mining techniques, text mining techniques, and advanced topics in both data and text mining. Each session will include a Python component, discussing the importance of Python and its libraries in handling various aspects of text and data mining. Students are not expected to know Python, rather they will be introduced to how Python can solve key issues so that they are aware of its capabilities. By the end of the course, participants will have a solid understanding of text and data mining concepts, be proficient in using Python for text and data mining tasks, and be able to apply these skills to real-world library applications and case studies.

# Learning Outcomes

1. Understanding of Data, Data Structures, and Complex Data Types
2. Understanding of the main types of Machine Learning and their Applications
3. Understanding of the key Python libraries for text and data mining
4. Understanding of the primary methods for performing text and data mining

# Calendar

| Date         | Title                                        | Description                                                                                                                               | Python Libraries                                                                                                                                                          |
|--------------|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 12 Oct 2023  | Introduction to Text and Data Mining         | This session introduces the course, delineating its structure, and shedding light on both basic and complex data types and structures. It also provides an overview of pertinent data mining concepts, highlighting the crucial role that Python plays in text and data mining. | - |
| 19 Oct 2023  | Data and Machine Learning                    | The class gives an overview of machine learning, emphasizing the indispensable role of data and introducing various types of machine learning. It addresses challenges in this field, discussing ethical considerations vital to machine learning. | [Scikit-Learn](https://scikit-learn.org/), [PyTorch and FastAI](https://www.pytorch.org/), [Tensorflow and Keras](https://www.tensorflow.org/) |
| 26 Oct 2023  | Data Pre-processing for Libraries            | This lecture delves into techniques essential for data cleaning, transformation, and reduction, crucial processes to prepare data for further analysis and use. | [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) |
| 2 Nov 2023   | Data Mining Techniques                       | The session explores a variety of data mining techniques including classification, clustering, and dimensionality reduction methods (PCA, t-SNE, UMAP, Ivis). These techniques will be discussed with special focus on library data. | [Scikit-Learn](https://scikit-learn.org/), [UMAP](https://umap-learn.readthedocs.io/), [HDBscan](https://hdbscan.readthedocs.io/) |
| 9 Nov 2023   | Text Processing for Library Data             | In this lecture, students will learn techniques for text cleaning, transformation, and representation, specifically applied to library text data. The session will also introduce various Natural Language Processing (NLP) libraries for text processing. | [NLTK](https://www.nltk.org/), [Spacy](https://spacy.io/) |
| 16 Nov 2023  | Text Mining Techniques                       | This session offers an in-depth discussion on various text mining techniques such as sentiment analysis, named entity recognition, topic modeling, and text classification. | [Gensim](https://radimrehurek.com/gensim/), [Spacy](https://spacy.io/), [BerTopic](https://maartengr.github.io/BERTopic/), [LeetTopic](https://github.com/wjbmattingly/LeetTopic) |
| 23 Nov 2023  | *Thanksgiving - No Class*                    | - | - |
| 30 Nov 2023  | Vector Databases and Semantic Searching      | This lecture provides an overview of vectors for both text and images and introduces best practices in the field. It covers machine learning models applicable to text and images, as well as introduces vector databases and semantic searching libraries. | [HuggingFace API](https://huggingface.co/api), [SentenceTransformers](https://www.sbert.net/), [Annoy](https://github.com/spotify/annoy), [Weaviate](https://weaviate.io/) |
| 7 Dec 2023   | Building Data Driven Applications            | The session provides an overview of various Python libraries and tools essential for building data-driven applications, illustrated through real-world case studies. | [Streamlit](https://streamlit.io/), [Gradio](https://gradio.app/), [Dash](https://plotly.com/dash/), [Plotly](https://plotly.com/python/) |

                                                             


# Event Sessions

## Thursday, October 12, 2023: Introduction to Text and Data Mining

- Introduction to the course and its structure
- Introduction to basic and complex data types and structures
- Overview of data mining concepts
- Role of Python in text and data mining

---

## Thursday, October 19, 2023: Data and Machine Learning

- Machine Learning Overview
- The Role of Data in Machine Learning
    - Rendering Images Numerically
    - Rendering Texts Numerically
    - Rendering Categorical Data Numerically
- Introduction to Types of Machine Learning:
    - Supervised Learning
    - Unsupervised Learning
    - Reinforcement Learning
- Challenges Machine Learning
- Ethical Considerations
    
### Resources shared:

- [Scikit-Learn](https://scikit-learn.org/)
- [PyTorch and FastAI](https://www.pytorch.org/)
- [Tensorflow and Keras](https://www.tensorflow.org/)

---

## Thursday, October 26, 2023: Data Pre-processing for Libraries

- Data Cleaning
    - Handling missing data
    - Working with noisy data
    - Working with Outliers
- Data Transformation
    - Data normalization
- Data Reduction
    - Dimensionality Reduction
    - Feature selection
    
### Resources shared:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---

## Thursday, November 2, 2023: Data Mining Techniques

- Classification
- Types of Classification
    - Binary Classification
    - Multiclass Classification
    - Multilabel Classification
    - Hierarchical Classification
- Resources for Open-Source Machine Learning Models
    
### Resources shared:
- [Hugging Face](https://huggingface.co/) - The platform where the machine learning community collaborates on models, datasets, and applications.
- [Hugging Face DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=we+enjoy+food) - An open source machine learning model. This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2. This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).
- [Hugging Face Dataset Card for "cardiffnlp/tweet_topic_multi"](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi?row=3) - This is the official repository of TweetTopic (["Twitter Topic Classification , COLING main conference 2022"](https://arxiv.org/abs/2209.09824)), a topic classification dataset on Twitter with 19 labels. Each instance of TweetTopic comes with a timestamp which distributes from September 2019 to August 2021. See [cardiffnlp/tweet_topic_single](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) for single label version of TweetTopic. The tweet collection used in TweetTopic is same as what used in [TweetNER7](https://huggingface.co/datasets/tner/tweetner7). The dataset is integrated in [TweetNLP](https://tweetnlp.org/) too.
- [Data Mining Teaching Tool](https://data-science-teaching.streamlit.app/) - App by Streamlit
---

## Thursday, November 9, 2023: Text Processing for Library Data

- Clustering
- Dimensionality Reduction
- Text Cleaning: removing punctuation, numbers, and special characters from library text data
- Text Transformation: tokenization, stemming, and lemmatization of library text data
- Text Representation: bag-of-words, TF-IDF, and word embeddings for library text data
- Python Component: Introduction to Natural Language Processing (NLP) libraries in Python like NLTK and spaCy.
    
### Resources shared:

- [NLTK](https://www.nltk.org/)
- [Spacy](https://spacy.io/)

---

## Thursday, November 16, 2023: Text Mining Techniques

- Sentiment Analysis
- Named Entity Recognition
- Topic Modeling
- Text Classification

### Resources shared:

- [Gensim](https://radimrehurek.com/gensim/)
- [Spacy](https://spacy.io/)
- [BerTopic](https://maartengr.github.io/BERTopic/)
- [LeetTopic](https://github.com/wjbmattingly/LeetTopic)

---

## Thursday, November 23, 2023: Thanksgiving - No Class

---

## Thursday, November 30, 2023: Vector Databases and Semantic Searching

- Refresher on Vectors for Text
- Vectors for Images
- Best Practices
- Machine Learning Models for Text
    - SentenceTransformers
- Machine Learning Models for Images
    - 
- Machine Learning Models for Images and Texts
    - CLIP
    
### Resources shared:

- [HuggingFace API](https://huggingface.co/api)
- [SentenceTransformers](https://www.sbert.net/)
- [Annoy](https://github.com/spotify/annoy)
- [Weaviate](https://weaviate.io/)

---

## Thursday, December 7, 2023: Building Data Driven Applications
- More Forthcoming...
- Overview of Python libraries and tools used for building data-driven applications with real-world case studies
    
### Resources shared:

- [Streamlit](https://streamlit.io/)
- [Gradio](https://gradio.app/)
- [Dash](https://plotly.com/dash/)
- [Plotly](https://plotly.com/python/)

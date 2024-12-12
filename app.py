# streamlit_app.py
import gensim
import streamlit as st
import re
import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
# Stop words
stop_words = set(stopwords.words('english'))

# Function to get WordNet POS tag for lemmatization
def get_wordnet_pos(word):
    """Map POS tag to WordNet POS tag"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN

# Load the saved Random Forest model
model_filename = "./RAF_Classifier.pkl"
with open(model_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Load the Word2Vec model
w2v_filename = "./w2v_model.pkl"
with open(w2v_filename, "rb") as file:
    w2v_model = pickle.load(file)

# Helper function to preprocess and vectorize the input text
def vectorize_text(text, model, embedding_dim=100):
    """
    Converts text into a feature vector by averaging Word2Vec embeddings.
    :param text: Input string
    :param model: Pre-trained Word2Vec model
    :param embedding_dim: Dimension of the embeddings
    :return: Numpy array of feature vector
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens if word not in stop_words]
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Streamlit App
st.title("Spam or Ham Classifier")
st.write("Enter a message below, and the model will predict whether it's **Spam** or **Ham**.")

# User Input
user_input = st.text_area("Type your message here:", "")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess and vectorize the input text
        input_vector = vectorize_text(user_input, w2v_model)

        # Predict using the loaded model
        prediction = loaded_model.predict([input_vector])

        # Display result
        if prediction[0] == 1:  # Assuming 1 represents spam
            st.error("The message is classified as **Spam**.")
        else:  # Assuming 0 represents ham
            st.success("The message is classified as **Ham**.")
    else:
        st.warning("Please enter a valid message.")

# Footer
st.write("---")
st.write("Created with ❤️ by Thirumal Reddy")

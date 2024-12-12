import streamlit as st
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
import pickle
from xgboost import XGBClassifier

# Load the trained XGBoost model and Word2Vec model
xgb_model = XGBClassifier()
xgb_model.load_model("xgboost_model.json")  # Save your model in the script using .save_model()

with open("word2vec_model.pkl", "rb") as file:
    word2vec_model = pickle.load(file)

# Initialize lemmatizer
lemma = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    review = [lemma.lemmatize(word) for word in review]  # Lemmatize
    return ' '.join(review)

# Average Word2Vec embedding
def avg_word2vec(doc):
    words = simple_preprocess(doc)
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv.index_to_key]
    if len(word_vectors) == 0:  # Handle empty documents
        return np.zeros(100)
    return np.mean(word_vectors, axis=0)

# Streamlit UI
st.title("Spam Classification App")

st.write("This app predicts whether a given message is spam or not using an XGBoost model.")

# Input from the user
input_text = st.text_area("Enter the text you want to classify:")

if st.button("Predict"):
    if input_text.strip():
        # Preprocess and vectorize the input
        processed_text = preprocess_text(input_text)
        feature_vector = avg_word2vec(processed_text).reshape(1, -1)  # Reshape for prediction

        # Predict
        prediction = xgb_model.predict(feature_vector)[0]
        prediction_label = "Spam" if prediction == 1 else "Not Spam"

        # Display result
        st.success(f"The message is classified as: {prediction_label}")
    else:
        st.warning("Please enter some text to classify.")

# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved Random Forest model
model_filename = "RAF_Classifier.pkl"
with open(model_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Load the vectorizer used during training
vectorizer_filename = "w2v_model.pkl"  # Save your vectorizer when training the model
with open(vectorizer_filename, "rb") as file:
    vectorizer = pickle.load(file)

# Streamlit App
st.title("Spam or Ham Classifier")
st.write("Enter a message below, and the model will predict whether it's **Spam** or **Ham**.")

# User Input
user_input = st.text_area("Type your message here:", "")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess the input text
        input_data = vectorizer.transform([user_input])  # Vectorize the input text

        # Predict using the loaded model
        prediction = loaded_model.predict(input_data)

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

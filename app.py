import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("SMS Spam Classifier")
st.write("Enter a message below to classify it as Spam or Not Spam:")

# Input from user
user_input = st.text_area("Message", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        # Preprocess and predict
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.success(f"The message is classified as: **{result}**")


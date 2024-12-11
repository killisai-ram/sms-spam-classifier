# SMS Spam Classifier

This repository contains a machine learning model that classifies SMS messages as either **Spam** or **Ham**. The model is built using scikit-learn and is integrated into a simple Streamlit web app for easy interaction.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [How to Use](#how-to-use)
    - [Clone the Repository](#1-clone-the-repository)
    - [Install Dependencies](#2-install-dependencies)
    - [Run the Streamlit App](#3-run-the-streamlit-app)
    - [Direct Model Usage](#4-direct-model-usage)
4. [How the Code Works](#how-the-code-works)
5. [Example Code](#example-code)
6. [License](#license)

## Introduction
This project uses a machine learning model to classify SMS messages into two categories:
- **Spam**: Unsolicited or unwanted messages.
- **Ham**: Legitimate messages.

The model uses two essential files:
- `spam_classifier.pkl`: The trained spam classification model.
- `vectorizer.pkl`: The vectorizer used to transform the input text into a format that the model can understand.

The repository also includes a **Streamlit** app (`app.py`) to easily classify messages through a user-friendly interface.

## Project Structure
The repository contains the following files:

sms-spam-classifier/ │ ├── app.py # Streamlit app for interactive classification ├── spam_classifier.pkl # Trained model for classifying spam/ham ├── vectorizer.pkl # Text vectorizer used with the model ├── requirements.txt # Python dependencies required to run the project └── README.md # Project documentation (this file)

bash
Copy code

## How to Use

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier
2. Install Dependencies
Next, install all the required libraries by running:

bash
Copy code
pip install -r requirements.txt
This will install the necessary dependencies, including Streamlit, scikit-learn, pandas, and numpy.

3. Run the Streamlit App
Once the dependencies are installed, you can run the Streamlit app using the following command:

bash
Copy code
streamlit run app.py
This will launch the Streamlit web app. Open your browser and navigate to http://localhost:8501 to interact with the app.

4. Direct Model Usage
If you prefer to use the model directly in your own Python script, here’s an example:

python
Copy code
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Example usage
message = "Free lottery ticket, claim now!"
input_vectorized = vectorizer.transform([message])  # Transform input text
prediction = model.predict(input_vectorized)       # Predict whether it's spam or ham

if prediction == 1:
    print("This message is Spam.")
else:
    print("This message is Ham.")
This will allow you to use the model programmatically to classify messages.

How the Code Works
Streamlit App (app.py):
The Streamlit app loads the spam_classifier.pkl model and vectorizer.pkl vectorizer. It allows users to input an SMS message, transforms the text using the vectorizer, and predicts whether the message is spam or ham using the trained model.

Model and Vectorizer:
spam_classifier.pkl: A scikit-learn model (such as Logistic Regression, Random Forest, etc.) trained on a dataset of SMS messages.
vectorizer.pkl: A TfidfVectorizer used to convert raw SMS text into numerical features that the model can process.
Prediction Process:
The user enters an SMS message in the web interface.
The message is transformed into numerical features using the TfidfVectorizer.
The transformed message is passed through the model to predict whether the message is Spam (1) or Ham (0).
Example Code
Here is an example of how the model and vectorizer can be used in a Python script to classify a new SMS message:

python
Copy code
import pickle

# Load model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Example SMS message
message = "Congrats! You've won a free vacation."

# Transform the message using the vectorizer
input_vectorized = vectorizer.transform([message])

# Predict using the model
prediction = model.predict(input_vectorized)

if prediction == 1:
    print("This message is Spam.")
else:
    print("This message is Ham.")
License
This project is licensed under the MIT License - see the LICENSE file for details.

vbnet
Copy code

### Key Styling Features:
- **Proper Code Block Styling**: Code blocks are wrapped in triple backticks (```), ensuring correct formatting and readability.
- **Headings**: Used headers for each section like `##`, `###`, etc., to ensure sections are clear and readable.
- **Code Inside Markdown**: Where you reference code or files, use **backticks** (single or triple) to ensure it's highlighted as code.

This will make your README file clean and properly formatted for GitHub or any markdown rende

import streamlit as st
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Instrument Analysis: Should You Buy It or not?')

    # Load the dataset
data = pd.read_csv('Musical_instruments_reviews.csv')
# Define input (reviewText) and output (overall) columns
x = data['reviewText']
y = data['overall']
ps = PorterStemmer()

    # Data cleaning function
def preprcess_data(content):
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    clean_data = re.sub('[^a-zA-Z]', ' ', content)
    lower_data = clean_data.lower()
    splited_data = lower_data.split()
    stemmed_data = [ps.stem(word) for word in splited_data]
    return ' '.join(stemmed_data)

    # Apply data cleaning function to the reviewText column
data['reviewText'] = data['reviewText'].apply(preprcess_data)
x = data['reviewText'].values
y = data['overall'].values
# Vectorize the cleaned text data
vector = TfidfVectorizer(stop_words='english')
vector.fit(x)
x = vector.transform(x)
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=100, stratify=y)
# Train a logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, Y_train)
# Calculate and display the accuracy scores
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))
# Optionally, allow user to input their own review for prediction
user_review = st.text_area("Enter a Instrument review to predict and get advice:")
if user_review:
        # Preprocess and vectorize the user's review
    cleaned_review = preprcess_data(user_review)
    review_vector = vector.transform([cleaned_review])
    # Predict the sentiment based on the user's review
    prediction = model.predict(review_vector)
    # Provide recommendation based on the predicted rating
    if prediction[0] >= 4:
        st.write(f"Predicted Rating: {prediction[0]}")
        st.success("You should buy it!")
    else:
            st.write(f"Predicted Rating: {prediction[0]}")
            st.error("You should not buy it.")

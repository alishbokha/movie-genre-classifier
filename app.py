import streamlit as st
import joblib
import re

vectorizer = joblib.load("tfidf_vectorizer.pkl")
genre_model = joblib.load("genre_classifier.pkl")
mlb = joblib.load("genre_encoder.pkl")

sentiment_tfidf = joblib.load("sentiment_tfidf.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")


def clean_text(text):
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(
    page_title="Movie NLP Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie NLP Classifier")
st.write(
    "Predict movie genres from a description and movie sentiment from a review."
)

st.subheader("🎬 Movie Description")

description = st.text_area(
    "Enter the movie description",
    placeholder="Example: A young wizard discovers a magical world and fights a powerful dark enemy.",
    height=150
)

st.subheader("⭐ Movie Review")

review = st.text_area(
    "Enter the movie review",
    placeholder="Example: This movie was amazing. The acting was fantastic and I really enjoyed the story.",
    height=150
)

if st.button("Predict", type="primary"):

    if description.strip():
        cleaned_description = clean_text(description)
        description_vector = vectorizer.transform([cleaned_description])
        genre_prediction = genre_model.predict(description_vector)
        predicted_genres = mlb.inverse_transform(genre_prediction)[0]

        st.subheader("🎬 Predicted Genres")

        if len(predicted_genres) == 0:
            st.info("No genre was predicted.")
        else:
            for genre in predicted_genres:
                st.success(genre)
    else:
        st.info("Enter a movie description to predict genres.")

    if review.strip():
        cleaned_review = clean_text(review)
        review_vector = sentiment_tfidf.transform([cleaned_review])
        sentiment_prediction = sentiment_model.predict(review_vector)[0]

        st.subheader("⭐ Movie Sentiment")

        if sentiment_prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
    else:
        st.info("Enter a movie review to predict sentiment.")

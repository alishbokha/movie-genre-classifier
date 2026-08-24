import streamlit as st
import joblib
import re

# Load the trained models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("genre_classifier.pkl")
mlb = joblib.load("genre_encoder.pkl")


def clean_text(text):
    """Apply the same basic cleaning used during model training."""
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Genre Classifier")
st.write(
    "Enter a movie description and the trained NLP model "
    "will predict its genres."
)

# User input
description = st.text_area(
    "Movie Description",
    placeholder="Example: A young wizard discovers a magical world "
                "and fights a powerful dark enemy.",
    height=180
)

if st.button("Predict Genres", type="primary"):
    if not description.strip():
        st.warning("Please enter a movie description.")
    else:
        # Clean input using the same preprocessing approach
        cleaned_description = clean_text(description)

        # Transform text using the saved TF-IDF vectorizer
        text_vector = vectorizer.transform([cleaned_description])

        # Predict the multi-label genres
        prediction = model.predict(text_vector)

        # Convert 0/1 labels into genre names
        predicted_genres = mlb.inverse_transform(prediction)[0]

        st.subheader("Predicted Genres")

        if len(predicted_genres) == 0:
            st.info("No genre was predicted.")
        else:
            for genre in predicted_genres:
                st.success(genre)

import streamlit as st
import joblib
import re

# Load genre models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
genre_model = joblib.load("genre_classifier.pkl")
mlb = joblib.load("genre_encoder.pkl")

# Load sentiment models
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
    "Choose what you want to analyze: movie genres or movie review sentiment."
)

# Let the user choose one function
option = st.radio(
    "What would you like to do?",
    [
        "🎬 Predict Movie Genres",
        "⭐ Analyze Movie Sentiment"
    ]
)

# -----------------------------
# Genre classification
# -----------------------------
if option == "🎬 Predict Movie Genres":

    st.subheader("🎬 Movie Genre Prediction")

    description = st.text_area(
        "Movie Description",
        placeholder=(
            "Example: A young wizard discovers a magical world "
            "and fights a powerful dark enemy."
        ),
        height=180
    )

    if st.button("Predict Genres", type="primary"):

        if not description.strip():
            st.warning("Please enter a movie description.")
        else:
            cleaned_description = clean_text(description)

            text_vector = vectorizer.transform(
                [cleaned_description]
            )

            prediction = genre_model.predict(text_vector)

            predicted_genres = mlb.inverse_transform(
                prediction
            )[0]

            st.subheader("🎬 Predicted Genres")

            if len(predicted_genres) == 0:
                st.info("No genre was predicted.")
            else:
                for genre in predicted_genres:
                    st.success(genre)


# -----------------------------
# Sentiment classification
# -----------------------------
else:

    st.subheader("⭐ Movie Review Sentiment")

    review = st.text_area(
        "Movie Review",
        placeholder=(
            "Example: This movie was amazing. "
            "The acting was fantastic and I really enjoyed the story."
        ),
        height=180
    )

    if st.button("Analyze Sentiment", type="primary"):

        if not review.strip():
            st.warning("Please enter a movie review.")
        else:
            cleaned_review = clean_text(review)

            review_vector = sentiment_tfidf.transform(
                [cleaned_review]
            )

            sentiment_prediction = sentiment_model.predict(
                review_vector
            )[0]

            st.subheader("⭐ Predicted Sentiment")

            if sentiment_prediction == 1:
                st.success("Positive 😊")
            else:
                st.error("Negative 😞")

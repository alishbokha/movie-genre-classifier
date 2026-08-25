import streamlit as st
import joblib
import re

# ============================================================
# LOAD MODELS
# ============================================================

genre_vectorizer = joblib.load("tfidf_vectorizer.pkl")
genre_model = joblib.load("genre_classifier.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

sentiment_vectorizer = joblib.load("sentiment_tfidf.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie AI Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #a8a8b3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .genre-card {
        display: inline-block;
        padding: 0.55rem 1rem;
        margin: 0.25rem;
        border-radius: 999px;
        background: #262936;
        border: 1px solid #444857;
        font-weight: 600;
    }

    .sentiment-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        background: #262936;
        border: 1px solid #444857;
        margin-top: 1rem;
    }

    .sentiment-positive {
        font-size: 2rem;
        font-weight: 800;
    }

    .sentiment-negative {
        font-size: 2rem;
        font-weight: 800;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie AI Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze movie genres and review sentiment using NLP</div>',
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

genre_tab, sentiment_tab = st.tabs(
    ["🎬 Genre Prediction", "⭐ Sentiment Analysis"]
)


# ============================================================
# GENRE TAB
# ============================================================

with genre_tab:

    st.markdown("### 🎬 Movie Genre Prediction")

    st.write(
        "Enter a movie description and the trained NLP model "
        "will predict its genres."
    )

    description = st.text_area(
        "Movie Description",
        placeholder=(
            "Example: A young wizard discovers a magical world "
            "and fights a powerful dark enemy."
        ),
        height=200,
        key="genre_description"
    )

    if st.button(
        "🎬 Predict Genres",
        type="primary",
        key="genre_button"
    ):

        if not description.strip():
            st.warning("Please enter a movie description.")
        else:

            cleaned_description = clean_text(description)

            description_vector = genre_vectorizer.transform(
                [cleaned_description]
            )

            prediction = genre_model.predict(
                description_vector
            )

            predicted_genres = genre_encoder.inverse_transform(
                prediction
            )[0]

            st.markdown(
                '<div class="result-title">🎬 Predicted Genres</div>',
                unsafe_allow_html=True
            )

            if len(predicted_genres) == 0:
                st.info("No genre was predicted.")
            else:

                cards = ""

                for genre in predicted_genres:
                    cards += (
                        f'<span class="genre-card">🎞️ {genre}</span>'
                    )

                st.markdown(cards, unsafe_allow_html=True)


# ============================================================
# SENTIMENT TAB
# ============================================================

with sentiment_tab:

    st.markdown("### ⭐ Movie Review Sentiment")

    st.write(
        "Enter a movie review and the trained NLP model "
        "will predict whether the sentiment is positive or negative."
    )

    review = st.text_area(
        "Movie Review",
        placeholder=(
            "Example: This movie was amazing. "
            "The acting was fantastic and I really enjoyed the story."
        ),
        height=200,
        key="sentiment_review"
    )

    if st.button(
        "⭐ Analyze Sentiment",
        type="primary",
        key="sentiment_button"
    ):

        if not review.strip():
            st.warning("Please enter a movie review.")
        else:

            cleaned_review = clean_text(review)

            review_vector = sentiment_vectorizer.transform(
                [cleaned_review]
            )

            sentiment_prediction = sentiment_model.predict(
                review_vector
            )[0]

            st.markdown(
                '<div class="result-title">⭐ Sentiment Result</div>',
                unsafe_allow_html=True
            )

            if sentiment_prediction == 1:

                st.markdown(
                    """
                    <div class="sentiment-card">
                        <div class="sentiment-positive">
                            😊 Positive
                        </div>
                        <p>The review has a positive sentiment.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="sentiment-card">
                        <div class="sentiment-negative">
                            😞 Negative
                        </div>
                        <p>The review has a negative sentiment.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built with Python, Scikit-learn, TF-IDF and Streamlit"
)

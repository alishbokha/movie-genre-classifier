# 🎬 Movie AI Analyzer

An NLP and Machine Learning project that analyzes movie-related text using **Natural Language Processing (NLP), TF-IDF, and Scikit-learn**.

The project contains two independent features:

- 🎬 **Movie Genre Classification**
- ⭐ **Movie Review Sentiment Analysis**

Both models are integrated into an interactive **Streamlit web application**.

---

## 🚀 Live Demo

🌐 **Try the Movie AI Analyzer:**

https://movie-genre-classifier-1.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to apply Natural Language Processing and Machine Learning to movie-related text.

The application allows users to independently choose between:

1. 🎬 Predicting movie genres from a movie description
2. ⭐ Predicting whether a movie review is Positive or Negative

The two tasks use separate trained models because they are different NLP classification problems.

---

# 🎬 Feature 1: Movie Genre Classification

The first model takes a **movie description** as input and predicts one or more movie genres.

Since a movie can belong to multiple genres, this is a **multi-label classification problem**.

### Example

**Input:**

> A detective investigates a series of mysterious murders and discovers a dangerous criminal organization.

**Possible predictions:**

- Crime
- Mystery
- Thriller
- Drama

### Supported Genres

The model supports **18 genres**:

- Action
- Adventure
- Animation
- Comedy
- Crime
- Drama
- Family
- Fantasy
- History
- Horror
- Music
- Mystery
- Romance
- Science Fiction
- TV Movie
- Thriller
- War
- Western

---

## 📊 Genre Dataset

The genre classification dataset contains:

- **9,634 movie samples**
- **7,707 training samples**
- **1,927 testing samples**
- **18 possible genres**

This is a **multi-label classification problem**, because one movie can belong to multiple genres.

For example:

text
Movie → Action + Adventure + Science Fiction

---

# ⭐ Feature 2: Movie Review Sentiment Analysis

The second model takes a **movie review** as input and predicts whether the review is:

- 😊 **Positive**
- 😞 **Negative**

### Example

**Input:**

> This movie was fantastic. The acting was excellent and I really enjoyed the story.

**Prediction:**

> 😊 Positive

---

## 📊 Sentiment Dataset

The sentiment classification model uses the **IMDb 50,000 movie review dataset**.

- **Original reviews:** 50,000
- **Positive reviews:** 25,000
- **Negative reviews:** 25,000
- **Duplicate reviews removed:** 418
- **Final unique reviews:** 49,582
- **Training reviews:** 39,665
- **Testing reviews:** 9,917

The dataset was divided into approximately **80% training data and 20% testing data**.

### Model

text
Movie Review
      ↓
 Text Cleaning
      ↓
     TF-IDF
      ↓
Logistic Regression
      ↓
Positive / Negative

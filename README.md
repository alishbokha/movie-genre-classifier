# 🎬 Movie AI Analyzer

An NLP-based Movie AI Analyzer that uses Machine Learning to analyze movie descriptions and movie reviews.

The application provides two independent features:

1. 🎬 **Movie Genre Classification**
2. ⭐ **Movie Review Sentiment Analysis**

The project is built with Python, Scikit-learn, TF-IDF and Streamlit.

---

## 🚀 Live Demo

🌐 **Try the application:**

https://movie-genre-classifier-1.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to apply Natural Language Processing (NLP) and Machine Learning to movie-related text.

The application can:

- Predict multiple genres from a movie description.
- Predict whether a movie review is Positive or Negative.
- Provide predictions through an interactive Streamlit web application.

The two tasks use separate trained models because they are different NLP classification problems.

---

# 🎬 Feature 1: Movie Genre Classification

The first model predicts one or more genres from a movie description.

### Example

**Input:**

> A detective investigates a series of mysterious murders and discovers a dangerous criminal organization.

**Possible prediction:**

- Crime
- Mystery
- Thriller
- Drama

### Genres

The model supports 18 genres:

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

```text
Movie → Action + Adventure + Science Fiction

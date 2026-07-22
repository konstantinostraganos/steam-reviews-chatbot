# 🎮 Steam Reviews Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about video games based on real Steam user reviews.

Built as part of the ITC 6110 — Natural Language Processing group project (MSc in Data Science, Deree — The American College of Greece, Spring 2026).

## Overview

The chatbot retrieves the most relevant reviews from a corpus of 10,000 Steam reviews using semantic similarity, then generates natural-language answers grounded in those reviews.

- **Retriever:** all-MiniLM-L6-v2 (Sentence Transformers)
- **Generator:** Llama 3.1 8B via Groq API
- **Frontend:** Streamlit
- **Reviews:** 10,000 stratified Steam reviews across 240+ games



## Features

- Ask free-form questions about games (e.g. "What are the best horror games?")
- Adjustable number of retrieved reviews (1–10)
- Transparency panel showing retrieved passages with similarity scores
- Quick-start example questions
- Dataset info sidebar

## Project Structure

```
├── app.py                                    # Streamlit application
├── requirements.txt                          # Python dependencies
├── steam_game_reviews_10_000_stratified.csv  # Review corpus
├── games_description.csv                     # Game metadata (genres)
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- A Groq API key ([get one here](https://console.groq.com/keys))

### Local Installation

```bash
git clone https://github.com/konstantinostraganos/steam-reviews-chatbot.git
cd steam-reviews-chatbot
pip install -r requirements.txt
```

### Configure API Key

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### Run

```bash
streamlit run app.py
```

## How It Works

1. **Encoding:** All 10,000 reviews are encoded into dense embeddings using all-MiniLM-L6-v2
2. **Retrieval:** The user's question is encoded with the same model and the top-k most similar reviews are retrieved via cosine similarity
3. **Generation:** Retrieved reviews are passed as context to Llama 3.1 8B (via Groq API), which generates an answer grounded only in the provided reviews

## Related Notebooks

The full NLP pipeline (preprocessing, embeddings, topic modelling, classification, XAI, and RAG development) is documented in the companion Colab notebooks submitted alongside this repository.

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://groq.com/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)



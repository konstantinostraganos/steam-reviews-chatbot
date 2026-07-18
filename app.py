import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
st.set_page_config(page_title="Steam Reviews Chatbot", page_icon="🎮", layout="wide")
 
@st.cache_resource
def load_data_and_model():
    df_reviews = pd.read_csv('steam_game_reviews_10_000_stratified.csv')
    df_games = pd.read_csv('games_description.csv')
 
    df_merged = df_reviews.merge(
        df_games[['name', 'genres']],
        left_on='game_name', right_on='name', how='left'
    ).drop(columns=['name'])
    df_merged['genres'] = df_merged['genres'].fillna('Unknown')
 
    sentences = []
    for _, row in df_merged.iterrows():
        sentence = (
            f'Game: "{row["game_name"]}" | '
            f'Genres: {row["genres"]} | '
            f'Recommendation: {row["recommendation"]} | '
            f'Review: {str(row["review"]).replace(chr(10), " ")}'
        )
        sentences.append(sentence)
 
    embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence_embeddings = embeddings_model.encode(sentences, show_progress_bar=True, batch_size=64)
 
    return df_merged, sentences, embeddings_model, sentence_embeddings
 
def generate_answer(prompt):
    """Call Groq API — free and fast"""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "API key not configured. Please add GROQ_API_KEY in Streamlit secrets."
 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content":
                "You are a knowledgeable gaming assistant. "
                "Answer the user's question based ONLY on the game reviews provided. "
                "Only mention games that appear in the provided reviews. "
                "Do not recommend games from outside the reviews. "
                "If the reviews don't contain enough information, say so clearly. "
                "Be concise and helpful. Always respond in English."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
 
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"API error: {str(e)}"
 
def rag_pipeline(question, top_k, embeddings_model, sentences, sentence_embeddings):
    if not question.strip():
        return "Please enter a question.", "", 0
 
    q_embedding = embeddings_model.encode([question])
    sims = cosine_similarity(q_embedding, sentence_embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    retrieved = [(sentences[i], float(sims[i])) for i in top_indices]
 
    if retrieved[0][1] < 0.4:
        return "This question doesn't seem to be related to games. Please ask something about Steam games!", "", 0
 
    context = "\n\n".join([
        f"Review {i+1}:\n{sent}"
        for i, (sent, sim) in enumerate(retrieved)
    ])
 
    prompt = (
        f"Here are relevant game reviews:\n\n{context}\n\n"
        f"Question: {question}"
    )
 
    t1 = time.time()
    answer = generate_answer(prompt)
    t2 = time.time()
 
    reviews_text = ""
    for i, (sent, sim) in enumerate(retrieved):
        reviews_text += f"**[{i+1}] Similarity: {sim:.3f}**\n{sent[:400]}\n\n---\n\n"
 
    return answer, reviews_text, round(t2 - t1, 2)
 
# === UI ===
st.title("🎮 Steam Reviews Chatbot")
st.markdown("Ask questions about games based on real Steam reviews!")
 
with st.spinner("Loading data and embedding model..."):
    df_merged, sentences, embeddings_model, sentence_embeddings = load_data_and_model()
 
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of reviews to retrieve", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.header("📊 Dataset Info")
    st.metric("Total Reviews", f"{len(sentences):,}")
    st.metric("Unique Games", df_merged['game_name'].nunique())
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown(
        "This chatbot uses **RAG** (Retrieval-Augmented Generation) "
        "to answer questions about games.\n\n"
        "**Retriever:** all-MiniLM-L6-v2\n\n"
        "**LLM:** Llama 3.1 8B (via Groq API)\n\n"
        "**Reviews:** 10,000 Steam reviews"
    )
 
st.markdown("**💡 Try these questions:**")
col1, col2, col3, col4 = st.columns(4)
if col1.button("Best fantasy game?", use_container_width=True):
    st.session_state['question'] = "What is the best fantasy game?"
if col2.button("CS2 worth buying?", use_container_width=True):
    st.session_state['question'] = "Is Counter-Strike 2 worth buying?"
if col3.button("Best horror games?", use_container_width=True):
    st.session_state['question'] = "What are the best horror games?"
if col4.button("Black Myth Wukong?", use_container_width=True):
    st.session_state['question'] = "What do players think about Black Myth Wukong?"
 
question = st.text_input(
    "🔍 Ask a question about games:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., What are the best RPG games?"
)
 
if st.button("Ask", type="primary") and question:
    with st.spinner("Searching reviews and generating answer..."):
        answer, reviews, gen_time = rag_pipeline(
            question, top_k, embeddings_model, sentences, sentence_embeddings
        )
    st.markdown("### 💬 Answer")
    st.success(answer)
    if gen_time > 0:
        st.caption(f"⏱️ Generated in {gen_time}s")
    st.markdown("### 📄 Retrieved Reviews")
    with st.expander("Click to see retrieved reviews", expanded=False):
        st.markdown(reviews)

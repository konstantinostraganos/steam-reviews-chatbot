import streamlit as st
import pandas as pd
import numpy as np
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
 
defdef rag_pipeline(question, top_k, embeddings_model, sentences, sentence_embeddings):
    if not question.strip():
        return "Please enter a question.", ""

    q_embedding = embeddings_model.encode([question])
    sims = cosine_similarity(q_embedding, sentence_embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    retrieved = [(sentences[i], float(sims[i])) for i in top_indices]

    # Check if question is relevant to games
    if retrieved[0][1] < 0.4:
        return "This question doesn't seem to be related to games. Please ask something about Steam games!", ""

    games_found = []
    for sent, sim in retrieved:
        game = sent.split('"')[1] if '"' in sent else "Unknown"
        if game not in games_found:
            games_found.append(game)

    rec_count = sum(1 for sent, _ in retrieved if 'Recommended' in sent and 'Not Recommended' not in sent)
    not_rec_count = sum(1 for sent, _ in retrieved if 'Not Recommended' in sent)

    answer = (
        f"Based on the top {top_k} most relevant reviews, the most related games are: "
        f"**{', '.join(games_found)}**.\n\n"
        f"Out of {top_k} retrieved reviews: "
        f"**{rec_count} Recommended** and **{not_rec_count} Not Recommended**.\n\n"
        f"Check the retrieved reviews below for detailed player opinions."
    )

    reviews_text = ""
    for i, (sent, sim) in enumerate(retrieved):
        reviews_text += f"**[{i+1}] Similarity: {sim:.3f}**\n{sent[:400]}\n\n---\n\n"

    return answer, reviews_text
 
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
        "**LLM:** Qwen2.5-0.5B-Instruct *(used in notebook)*\n\n"
        "**Reviews:** 10,000 Steam reviews\n\n"
        "The retrieval component runs live here. "
        "Full LLM generation is demonstrated in the project notebook."
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
    with st.spinner("Searching reviews..."):
        answer, reviews = rag_pipeline(
            question, top_k, embeddings_model, sentences, sentence_embeddings
        )
    st.markdown("### 💬 Answer")
    st.success(answer)
    st.markdown("### 📄 Retrieved Reviews")
    with st.expander("Click to see retrieved reviews", expanded=True):
        st.markdown(reviews)

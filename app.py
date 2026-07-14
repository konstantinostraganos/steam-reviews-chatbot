import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Steam Reviews Chatbot",
    page_icon="🎮",
    layout="wide"
)

# ============================================================
# Load models & data (cached so it runs only once)
# ============================================================
@st.cache_resource
def load_models():
    """Load embedding model and LLM"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding model
    embeddings_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device=device
    )

    # LLM
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return embeddings_model, tokenizer, llm, device

@st.cache_resource
def load_data():
    """Load and merge datasets, build sentences and embeddings"""
    df_reviews = pd.read_csv('steam_game_reviews_10_000_stratified.csv')
    df_games = pd.read_csv('games_description.csv')

    df_merged = df_reviews.merge(
        df_games[['name', 'genres']],
        left_on='game_name',
        right_on='name',
        how='left'
    ).drop(columns=['name'])
    df_merged['genres'] = df_merged['genres'].fillna('Unknown')

    # Build sentences
    sentences = []
    for _, row in df_merged.iterrows():
        sentence = (
            f'Game: "{row["game_name"]}" | '
            f'Genres: {row["genres"]} | '
            f'Recommendation: {row["recommendation"]} | '
            f'Review: {str(row["review"]).replace(chr(10), " ")}'
        )
        sentences.append(sentence)

    return df_merged, sentences

@st.cache_resource
def encode_sentences(sentences):
    """Encode all sentences"""
    embeddings_model, _, _, _ = load_models()
    sentence_embeddings = embeddings_model.encode(
        sentences,
        show_progress_bar=True,
        batch_size=64
    )
    return sentence_embeddings

# ============================================================
# RAG Pipeline
# ============================================================
def rag_pipeline(question, top_k, embeddings_model, tokenizer, llm, device, sentences, sentence_embeddings):

    # --- Retrieve ---
    q_embedding = embeddings_model.encode([question])
    sims = cosine_similarity(q_embedding, sentence_embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    retrieved = [(sentences[i], float(sims[i])) for i in top_indices]

    # --- Prompt ---
    context = "\n\n".join([
        f"Review {i+1}:\n{sent}"
        for i, (sent, sim) in enumerate(retrieved)
    ])

    messages = [
        {"role": "system", "content":
            "You are a knowledgeable gaming assistant. "
            "Answer the user's question based ONLY on the game reviews provided below. "
            "Only mention games that appear in the provided reviews. "
            "Do not recommend games from outside the reviews. "
            "If the reviews don't contain enough information to answer, say so clearly. "
            "Do not make up information. Be concise and helpful. "
            "Always respond in English."
        },
        {"role": "user", "content":
            f"Here are relevant game reviews:\n\n{context}\n\n"
            f"Question: {question}"
        }
    ]

    # --- Generate ---
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t1 = time.time()
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    t2 = time.time()

    input_length = inputs['input_ids'].shape[1]
    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return {
        'question': question,
        'answer': answer,
        'retrieved': retrieved,
        'time': round(t2 - t1, 2)
    }

# ============================================================
# UI
# ============================================================
st.title("🎮 Steam Reviews Chatbot")
st.markdown("Ask questions about games based on real Steam reviews!")

# Load everything
with st.spinner("Loading models and data..."):
    embeddings_model, tokenizer, llm, device = load_models()
    df_merged, sentences = load_data()
    sentence_embeddings = encode_sentences(sentences)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of reviews to retrieve", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.header("📊 Dataset Info")
    st.metric("Total Reviews", len(sentences))
    st.metric("Unique Games", df_merged['game_name'].nunique())
    st.metric("Device", device.upper())
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown(
        "This chatbot uses **RAG** (Retrieval-Augmented Generation) "
        "to answer questions about games based on **10,000 Steam reviews**.\n\n"
        "**Retriever:** all-MiniLM-L6-v2\n\n"
        "**LLM:** Qwen2.5-0.5B-Instruct"
    )

# Example questions
st.markdown("**Try these questions:**")
example_cols = st.columns(4)
examples = [
    "What is the best fantasy game?",
    "Is Counter-Strike 2 worth buying?",
    "Best horror games?",
    "What do players think about Black Myth Wukong?"
]
for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state['question'] = example

# Input
question = st.text_input(
    "🔍 Ask a question about games:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., What are the best RPG games?"
)

# Generate
if st.button("Ask", type="primary") and question:
    with st.spinner("Searching reviews and generating answer..."):
        result = rag_pipeline(
            question, top_k,
            embeddings_model, tokenizer, llm, device,
            sentences, sentence_embeddings
        )

    # Answer
    st.markdown("### 💬 Answer")
    st.success(result['answer'])
    st.caption(f"⏱️ Generated in {result['time']}s")

    # Retrieved reviews
    st.markdown("### 📄 Retrieved Reviews")
    for i, (sent, sim) in enumerate(result['retrieved']):
        with st.expander(f"Review {i+1} — Similarity: {sim:.3f}"):
            st.text(sent[:500])

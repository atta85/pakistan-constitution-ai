import json
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

APP_TITLE = "Constitution of Pakistan AI Assistant"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 6
MAX_TOP_K = 12
DEFAULT_SIMILARITY_THRESHOLD = 0.25
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "constitution.faiss"
METADATA_PATH = BASE_DIR / "metadata.json"
CONFIG_PATH = BASE_DIR / "config.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading constitutional knowledge base...")
def load_resources():
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    index = faiss.read_index(str(FAISS_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, embedding_model

def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error(
            "Groq API key is not configured. Add GROQ_API_KEY "
            "to Streamlit Community Cloud Secrets."
        )
        st.stop()

    return Groq(api_key=api_key)

def clean_query(query):
    return re.sub(r"\s+", " ", query.strip())

def extract_article_number(query):
    patterns = [
        r"\barticle\s+(\d+[A-Za-z]?)\b",
        r"\bart\.?\s+(\d+[A-Za-z]?)\b",
        r"\barticle\s+no\.?\s*(\d+[A-Za-z]?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def create_query_embedding(embedding_model, query):
    query_for_embedding = (
        "Represent this sentence for searching relevant passages: " + query
    )
    embedding = embedding_model.encode(
        [query_for_embedding],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embedding, dtype="float32")

def retrieve_documents(
    index,
    metadata,
    embedding_model,
    query,
    top_k=DEFAULT_TOP_K,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
):
    chunks = metadata["chunks"]
    query_embedding = create_query_embedding(embedding_model, query)

    search_k = min(max(top_k * 3, 20), index.ntotal)
    scores, indices = index.search(query_embedding, search_k)

    article_number = extract_article_number(query)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        score = float(score)
        if score < similarity_threshold:
            continue

        chunk = chunks[int(idx)]
        article_match = False

        if article_number:
            article_text = str(chunk.get("article", ""))
            if article_text:
                article_match = (
                    re.search(
                        rf"\b{re.escape(article_number)}\b",
                        article_text,
                    )
                    is not None
                )

        results.append(
            {
                "score": score,
                "article_match": article_match,
                "chunk": chunk,
            }
        )

    if article_number:
        results.sort(
            key=lambda x: (x["article_match"], x["score"]),
            reverse=True,
        )
    else:
        results.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    seen_texts = set()

    for result in results:
        text = result["chunk"]["text"].strip()
        signature = re.sub(r"\s+", " ", text.lower())[:500]

        if signature in seen_texts:
            continue

        seen_texts.add(signature)
        selected.append(result)

        if len(selected) >= top_k:
            break

    return selected

def build_context(results):
    blocks = []

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]
        block = f"""
SOURCE {i}
Article: {chunk.get("article", "Not specified")}
Chapter: {chunk.get("chapter", "Not specified")}
PDF page: {chunk.get("page_start", "Unknown")}-{chunk.get("page_end", chunk.get("page_start", "Unknown"))}
Retrieval similarity: {result["score"]:.3f}

TEXT:
{chunk["text"]}
"""
        blocks.append(block.strip())

    return "\n\n".join(blocks)

SYSTEM_PROMPT = """
You are the Constitution of Pakistan AI Assistant.

Your primary task is to answer questions using the supplied retrieved
passages from the Constitution of the Islamic Republic of Pakistan.

GROUNDING RULES:
1. Use the supplied constitutional passages as your primary authority.
2. Do NOT invent an Article, clause, constitutional power, right,
   institution, procedure, date, or quotation.
3. If the retrieved material does not adequately answer the question,
   explicitly say that the available constitutional text retrieved
   for this question is insufficient.
4. Distinguish between what the Constitution expressly states and
   explanatory information.
5. When possible, identify the relevant Article.
6. Do not claim a provision exists merely because it would be logical
   or commonly discussed elsewhere.
7. Do not treat general legal commentary, political commentary,
   court interpretation, or personal opinion as constitutional text
   unless such material is explicitly provided as a source.
8. For exact quotations, reproduce only the relevant short passage
   from the supplied source and identify the Article/page.
9. If the question is ambiguous, explain the ambiguity.
10. Do not fabricate legal citations.
11. This application is an information and research assistant, not
    a substitute for professional legal advice.

ANSWER STYLE:
- Be clear and academically professional.
- Prefer concise answers unless detail is needed.
- Use headings and bullet points where useful.
- Mention relevant Article numbers.
- Explain constitutional language in plain English when useful.
- At the end, provide a short Sources section identifying retrieved
  Articles/pages used.
"""

def generate_answer(
    groq_client,
    model_name,
    question,
    context,
    conversation_history,
):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for message in conversation_history[-6:]:
        if message["role"] in ["user", "assistant"]:
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED CONSTITUTIONAL CONTEXT:
{context}

Answer the user's question strictly using the retrieved constitutional
material. If the retrieved material is insufficient, say so clearly.
Do not fill missing constitutional information with guesses.
"""
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=1800,
    )
    return response.choices[0].message.content

with st.sidebar:
    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
    )

    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.05,
        max_value=0.60,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        step=0.05,
    )

    st.divider()
    st.subheader("Groq Model")

    groq_model = st.text_input(
        "Model ID",
        value=DEFAULT_GROQ_MODEL,
        help="Use a currently available Groq model ID.",
    )

    st.divider()
    st.subheader("Knowledge Base")

    try:
        index, metadata, embedding_model = load_resources()
        st.success("Knowledge base loaded")
        st.write(f"**Vectors:** {index.ntotal:,}")
        st.write(f"**Chunks:** {len(metadata['chunks']):,}")
        st.write(f"**Embedding:** {EMBEDDING_MODEL}")
    except Exception as e:
        st.error(f"Knowledge base error: {e}")
        st.stop()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    '<div class="main-title">🇵🇰 Constitution of Pakistan AI Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">An AI-powered Retrieval-Augmented Generation '
    "assistant grounded in the Constitution of the Islamic Republic of Pakistan."
    "</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Retrieved constitutional sources"):
                for source in message["sources"]:
                    chunk = source["chunk"]
                    st.markdown(
                        f"**{chunk.get('article', 'Not specified')}** · "
                        f"{chunk.get('chapter', 'Not specified')} · "
                        f"PDF page {chunk.get('page_start', 'Unknown')} · "
                        f"similarity {source['score']:.3f}"
                    )
                    st.caption(chunk["text"])
                    st.divider()

question = st.chat_input(
    "Ask a question about the Constitution of Pakistan..."
)

if question:
    question = clean_query(question)

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        status.info("🔎 Searching the constitutional knowledge base...")

        results = retrieve_documents(
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not results:
            answer = (
                "I could not retrieve sufficiently relevant constitutional "
                "passages to answer this question reliably from the available "
                "Constitution dataset. Please try rephrasing the question or "
                "specifying the relevant Article."
            )
            status.empty()
            st.warning(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
            st.stop()

        context = build_context(results)
        status.info(
            f"📚 Retrieved {len(results)} relevant passages. "
            "Generating grounded answer..."
        )

        try:
            groq_client = get_groq_client()
            previous_history = st.session_state.messages[:-1]

            answer = generate_answer(
                groq_client=groq_client,
                model_name=groq_model,
                question=question,
                context=context,
                conversation_history=previous_history,
            )
        except Exception as e:
            answer = (
                "An error occurred while contacting the Groq language "
                f"model:\n\n`{str(e)}`"
            )

        status.empty()
        st.markdown(answer)

        with st.expander("📚 Retrieved constitutional sources"):
            for source in results:
                chunk = source["chunk"]
                st.markdown(
                    f"**{chunk.get('article', 'Not specified')}** · "
                    f"{chunk.get('chapter', 'Not specified')} · "
                    f"PDF page {chunk.get('page_start', 'Unknown')} · "
                    f"similarity {source['score']:.3f}"
                )
                st.caption(chunk["text"])
                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )

st.divider()
st.caption(
    "Constitution of Pakistan AI Assistant · "
    "RAG · FAISS + BGE embeddings + Groq"
)

# app.py
import streamlit as st
from crawler import crawl
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import store_embeddings, get_collection_stats
from rag import ask

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide"
)

# ─── Session State ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # chat display messages
if "history" not in st.session_state:
    st.session_state.history = []        # RAG conversation memory
if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = []  # track ingested URLs


# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Pipeline")
    st.markdown("---")

    st.subheader("📥 Ingest a Website")
    url = st.text_input("Enter URL", placeholder="https://example.com")
    depth = st.slider("Crawl Depth", min_value=0, max_value=3, value=1)
    chunk_size = st.slider("Chunk Size (chars)", min_value=256, max_value=1024, step=256, value=512)

    if st.button("🚀 Ingest", use_container_width=True):
        if not url:
            st.error("Please enter a URL.")
        else:
            with st.spinner("Crawling..."):
                pages = crawl(url, max_depth=depth)
                st.success(f"✅ Crawled {len(pages)} pages")

            with st.spinner("Chunking..."):
                chunks = chunk_pages(pages, max_chars=chunk_size)
                st.success(f"✅ Created {len(chunks)} chunks")

            with st.spinner("Embedding..."):
                embedded = embed_chunks(chunks)
                st.success(f"✅ Embedded {len(embedded)} chunks")

            with st.spinner("Storing in Pinecone..."):
                store_embeddings(embedded)
                st.success(f"✅ Stored in Pinecone!")

            st.session_state.ingested_urls.append(url)

    st.markdown("---")

    # Show ingested URLs
    if st.session_state.ingested_urls:
        st.subheader("🌐 Ingested URLs")
        for u in st.session_state.ingested_urls:
            st.markdown(f"- {u}")

    st.markdown("---")

    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# ─── Main Chat Area ────────────────────────────────────────
st.title("💬 Chat with your Data")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the ingested content..."):
    if not st.session_state.ingested_urls:
        st.warning("⚠️ Please ingest a website first using the sidebar.")
    else:
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, updated_history = ask(
                    prompt,
                    history=st.session_state.history
                )
                st.session_state.history = updated_history
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
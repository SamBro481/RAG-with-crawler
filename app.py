# app.py
import streamlit as st
from crawler import crawl
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import store_embeddings
from rag import ask
from auth import login, sign_up, save_crawled_site, get_crawled_sites
import hashlib
from auth import login, sign_up, save_crawled_site, get_crawled_sites, delete_crawled_site
from vector_store import store_embeddings, delete_namespace

def get_namespace(user_id: str, url: str) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"user_{user_id[:8]}_{url_hash}"

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide"
)

if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "active_site" not in st.session_state:
    st.session_state.active_site = None


def show_auth():

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🔍 RAG Pipeline</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Chat with any website</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login", use_container_width=True, key="login_btn"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    result = login(username, password)
                    if result["success"]:
                        st.session_state.user = result
                        st.rerun()
                    else:
                        st.error(result["error"])

        with tab2:
            username = st.text_input("Username", key="signup_user")
            password = st.text_input("Password", type="password", key="signup_pass")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign Up", use_container_width=True, key="signup_btn"):
                if not username or not password or not confirm:
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords don't match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    result = sign_up(username, password)
                    if result["success"]:
                        st.session_state.user = result
                        st.rerun()
                    else:
                        st.error(result["error"])


def show_app():
    user = st.session_state.user

    with st.sidebar:
        st.title("🔍 RAG Pipeline")
        st.markdown(f"👤 Logged in as **{user['username']}**")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.active_site = None
            st.rerun()

        st.markdown("---")

        st.subheader("Ingest a Website")
        url = st.text_input("Enter URL", placeholder="https://example.com")
        depth = st.slider("Crawl Depth", min_value=0, max_value=3, value=1)
        chunk_size = st.slider("Chunk Size (chars)", min_value=256, max_value=1024, step=256, value=512)

        if st.button("Ingest", use_container_width=True):
            if not url:
                st.error("Please enter a URL.")
            else:
                try:
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
                        namespace = get_namespace(user["user_id"], url)
                        store_embeddings(embedded, namespace=namespace)
                        st.success(f"✅ Stored in Pinecone!")

                    save_crawled_site(user["user_id"], url, len(pages), len(chunks))
                    st.session_state.active_site = url
                    st.rerun()

                except ValueError as e:
                    st.error(f"{e}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        st.markdown("---")

        st.subheader("Your Crawled Sites")
        crawled_sites = get_crawled_sites(user["user_id"])

        if not crawled_sites:
            st.info("No sites crawled yet.")
        else:
            for site in crawled_sites:
                st.markdown(f"🔗 `{site['url'][:35]}...`")
                st.caption(f"{site['pages_count']} pages · {site['chunks_count']} chunks")
    
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("💬 Chat", key=f"chat_{site['id']}", use_container_width=True):
                        st.session_state.active_site = site["url"]
                        st.session_state.messages = []
                        st.session_state.history = []
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{site['id']}", use_container_width=True):

                        ns = get_namespace(user["user_id"], site["url"])
                        delete_namespace(ns)

                        delete_crawled_site(user["user_id"], site["url"])

                        if st.session_state.active_site == site["url"]:
                            st.session_state.active_site = None
                            st.session_state.messages = []
                            st.session_state.history = []
                        st.rerun()
    
                st.markdown("<br>", unsafe_allow_html=True)
                
            
            
            
                

        if st.session_state.active_site:
            st.markdown("---")
            st.success(f"Chatting with:\n{st.session_state.active_site[:50]}...")
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.history = []
                st.rerun()

    st.title("Chat with your Data")

    if not st.session_state.active_site:
        st.info("Ingest a website or select a previously crawled site from the sidebar to start chatting.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- [{source}]({source})")

    if prompt := st.chat_input("Ask a question about the ingested content..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, updated_history, sources = ask(
                    prompt,
                    history=st.session_state.history,
                    namespace=get_namespace(user["user_id"], st.session_state.active_site)
                )
                st.session_state.history = updated_history
                st.markdown(answer)
                
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if st.session_state.user is None:
    show_auth()
else:
    show_app()
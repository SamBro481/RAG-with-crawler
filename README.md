# RAG Pipeline 🔍

A full Retrieval-Augmented Generation (RAG) pipeline with a web crawler, semantic search, and conversational memory.

#### Visualise it Here: https://webcrawler-rag.streamlit.app

## Stack
- **Crawler** — requests + BeautifulSoup
- **Embeddings** — OpenAI text-embedding-3-small
- **Vector DB** — Pinecone
- **LLM** — GPT-4o-mini
- **Frontend** — Streamlit

## Setup

1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

2. Install dependencies
pip install -r requirements.txt

3. Add your API keys
cp .env.example .env, then fill in your keys

4. Run the app
streamlit run app.py

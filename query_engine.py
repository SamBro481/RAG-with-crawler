# query_engine.py
from openai import OpenAI
from vector_store import search_embeddings

client = OpenAI(api_key="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")
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def search(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embed_query(query)
    return search_embeddings(query_vector, top_k=top_k)


def display_results(query: str, matches: list[dict]):
    print(f"\n🔍 Query: {query}")
    print(f"{'─' * 60}")
    for i, match in enumerate(matches):
        print(f"\n📄 Match {i + 1} (similarity: {match['similarity']})")
        print(f"🔗 Source: {match['url']}")
        print(f"📝 Text: {match['text']}")
        print(f"{'─' * 60}")
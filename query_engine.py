import os
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import search_embeddings

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def search(query: str, top_k: int = 3, namespace: str = "default") -> list[dict]:
    query_vector = embed_query(query)
    return search_embeddings(query_vector, top_k=top_k, namespace=namespace)
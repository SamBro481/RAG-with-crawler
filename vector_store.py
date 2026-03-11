# vector_store.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-pipeline"
DIMENSION = 1536  # text-embedding-3-small dimension


def get_or_create_index():
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Index created!")
    
    return pc.Index(INDEX_NAME)


def store_embeddings(embedded_chunks: list[dict]):
    index = get_or_create_index()

    vectors = []
    for chunk in embedded_chunks:
        chunk_id = f"{chunk['url']}__chunk{chunk['chunk_index']}"
        vectors.append({
            "id": chunk_id,
            "values": chunk["embedding"],
            "metadata": {
                "url": chunk["url"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]        # store text in metadata for retrieval
            }
        })

    # Pinecone recommends upserting in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"✅ Stored {len(vectors)} chunks in Pinecone")


def get_collection_stats():
    index = get_or_create_index()
    stats = index.describe_index_stats()
    print(f"📦 Pinecone index has {stats['total_vector_count']} chunks")


def search_embeddings(query_vector: list[float], top_k: int = 3) -> list[dict]:
    index = get_or_create_index()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = []
    for match in results["matches"]:
        matches.append({
            "text": match["metadata"]["text"],
            "url": match["metadata"]["url"],
            "chunk_index": match["metadata"]["chunk_index"],
            "similarity": round(match["score"], 4)
        })

    return matches
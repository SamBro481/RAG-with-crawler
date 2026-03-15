import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-pipeline"
DIMENSION = 1536


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
        print(f"Index created!")

    return pc.Index(INDEX_NAME)


def store_embeddings(embedded_chunks: list[dict], namespace: str = "default"):
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
                "text": chunk["text"]
            }
        })

    batches = [vectors[i:i + 100] for i in range(0, len(vectors), 100)]
    print(f"Uploading {len(vectors)} vectors in {len(batches)} parallel batches...")

    def upsert_batch(batch):
        index.upsert(vectors=batch, namespace=namespace)

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(upsert_batch, batches))

    print(f"✅ Stored {len(vectors)} chunks in namespace '{namespace}'")


    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)  # ← namespace added

    print(f"Stored {len(vectors)} chunks in namespace '{namespace}'")


def get_collection_stats(namespace: str = "default"):
    index = get_or_create_index()
    stats = index.describe_index_stats()
    count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    print(f"Namespace '{namespace}' has {count} chunks")


def delete_namespace(namespace: str):
    try:
        index = get_or_create_index()
        index.delete(delete_all=True, namespace=namespace)
        print(f"🗑️ Deleted namespace '{namespace}'")
    except Exception as e:
        print(f"⚠️ Namespace '{namespace}' not found in Pinecone, skipping...")

def search_embeddings(query_vector: list[float], top_k: int = 3, namespace: str = "default") -> list[dict]:
    index = get_or_create_index()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace     
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
# embedder.py
from openai import OpenAI

client = OpenAI(api_key="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")

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    print(f"Embedding {len(chunks)} chunks in one batch...")
    
    texts = [chunk["text"] for chunk in chunks]
    
    response = client.embeddings.create(
        input=texts,          # send all at once
        model=EMBEDDING_MODEL
    )
    
    embedded = []
    for i, chunk in enumerate(chunks):
        embedded.append({
            **chunk,
            "embedding": response.data[i].embedding
        })
    
    return embedded


if __name__ == "__main__":
    test_chunks = [
        {"url": "https://example.com", "chunk_index": 0, "text": "Machine learning allows systems to learn from data without being explicitly programmed."},
        {"url": "https://example.com", "chunk_index": 1, "text": "Deep learning uses neural networks with many layers to recognize images and speech."},
    ]

    embedded = embed_chunks(test_chunks)

    for chunk in embedded:
        print(f"\nChunk {chunk['chunk_index']}")
        print(f"Text: {chunk['text'][:60]}...")
        print(f"Vector length: {len(chunk['embedding'])}")
        print(f"First 5 values: {chunk['embedding'][:5]}")

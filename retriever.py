import os
import cohere
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from query_engine import embed_query
from vector_store import search_embeddings
from intent import detect_intent

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def hybrid_search(
    query: str,
    namespace: str = "default",
    top_k: int = 6,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> list[dict]:

    query_vector = embed_query(query)
    semantic_results = search_embeddings(query_vector, top_k=top_k, namespace=namespace)

    if not semantic_results:
        return []

    corpus = [chunk["text"] for chunk in semantic_results]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_normalized = bm25_scores / bm25_max

    for i, chunk in enumerate(semantic_results):
        semantic_score = chunk["similarity"]
        bm25_score = float(bm25_scores_normalized[i])
        chunk["hybrid_score"] = round(
            semantic_weight * semantic_score + bm25_weight * bm25_score, 4
        )

    return sorted(semantic_results, key=lambda x: x["hybrid_score"], reverse=True)


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    if not chunks:
        return []

    docs = [chunk["text"] for chunk in chunks]

    results = co.rerank(
        query=query,
        documents=docs,
        top_n=top_k,
        model="rerank-english-v3.0"
    )

    reranked = []
    for r in results.results:
        chunk = chunks[r.index]
        chunk["rerank_score"] = round(r.relevance_score, 4)
        reranked.append(chunk)

    return reranked


def retrieve(query: str, namespace: str = "default", top_k: int = 3, history: list[dict] = []) -> tuple[list[dict], dict]:
    print(f"\n🔎 Retrieving for: {query}")


    intent_data = detect_intent(query, history)
    adjusted_top_k = intent_data["top_k"] 

    print(f"📌 Using top_k={adjusted_top_k} for intent '{intent_data['intent']}'")

    hybrid_results = hybrid_search(
        query,
        namespace=namespace,
        top_k=adjusted_top_k * 2  
    )
    print(f"Hybrid search returned {len(hybrid_results)} chunks")

    final_results = rerank(query, hybrid_results, top_k=adjusted_top_k)
    print(f"Re-ranked to top {len(final_results)} chunks")

    return final_results, intent_data
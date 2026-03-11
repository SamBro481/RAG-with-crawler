# rag.py
from openai import OpenAI
from query_engine import search

client = OpenAI(api_key="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")

# How many past messages to remember
MEMORY_SIZE = 6  # 3 exchanges (user + assistant pairs)


def rewrite_query(query: str, history: list[dict]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer. "
                "Given the conversation history and the user's latest question, "
                "rewrite it into a short keyword-focused search query "
                "that will retrieve the most relevant text from a vector database. "
                "Return only the rewritten query, nothing else."
            )
        },
        *history,  # include conversation history so it understands context
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"🔄 Rewritten query: {rewritten}")
    return rewritten


def generate_answer(query: str, chunks: list[dict], history: list[dict]) -> str:
    if not chunks:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n".join([
        f"Source: {c['url']}\n{c['text']}"
        for c in chunks
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the provided context. If the answer isn't in the context, say "
                "'I couldn't find that information in the provided sources.' "
                "Be concise and precise."
            )
        },
        *history,   # inject last N messages for context
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content.strip()


def ask(query: str, history: list[dict] = [], top_k: int = 3) -> tuple[str, list[dict]]:
    print(f"\n🔍 Original query: {query}")

    # Keep only last N messages
    trimmed_history = history[-MEMORY_SIZE:]

    # Step 1 — rewrite query using history for better retrieval
    rewritten = rewrite_query(query, trimmed_history)

    # Step 2 — retrieve relevant chunks
    chunks = search(rewritten, top_k=top_k)

    # Step 3 — generate answer with history as context
    answer = generate_answer(query, chunks, trimmed_history)

    print(f"\n💬 Answer: {answer}")
    print(f"\n📚 Sources:")
    for c in chunks:
        print(f"  - {c['url']} (chunk {c['chunk_index']})")

    # Step 4 — update history and return it
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history


if __name__ == "__main__":
    history = []
    while True:
        query = input("\nAsk a question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        answer, history = ask(query, history)
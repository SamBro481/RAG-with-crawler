import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MEMORY_SIZE = 6


def rewrite_query(query: str, history: list[dict], intent: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer. "
                "Given the conversation history, the user's intent, and their question, "
                "rewrite it into a short keyword-focused search query. "
                "Strip conversational words like 'what did', 'can you tell me', 'what was'. "
                f"The user's intent is: {intent}. "
                "For summarization intent, keep the query broad. "
                "For fact_lookup intent, focus on the specific entity or value. "
                "For comparison intent, include both topics being compared. "
                "Return only the rewritten query, nothing else."
            )
        },
        *history,
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"🔄 Rewritten query: {rewritten}")
    return rewritten


def generate_answer(query: str, chunks: list[dict], history: list[dict], intent: str) -> str:
    if not chunks:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n".join([
        f"Source: {c['url']}\n{c['text']}"
        for c in chunks
    ])

    # Adjust system prompt based on intent
    intent_instructions = {
        "summarization": "Provide a comprehensive summary covering all key points from the context.",
        "fact_lookup": "Extract and state the specific fact precisely and concisely.",
        "explanation": "Explain clearly and thoroughly using the context provided.",
        "comparison": "Compare the topics systematically, highlighting similarities and differences.",
        "instruction": "List the steps or instructions clearly in order.",
        "opinion": "Provide a balanced recommendation based only on the context."
    }

    intent_instruction = intent_instructions.get(intent, "Answer concisely and precisely.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the provided context. If the answer isn't in the context, say "
                "'I couldn't find that information in the provided sources.' "
                f"{intent_instruction}"
            )
        },
        *history,
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


def ask(query: str, history: list[dict] = [], top_k: int = 3, namespace: str = "default") -> tuple[str, list[dict], list[str]]:
    print(f"\n🔍 Original query: {query}")

    trimmed_history = history[-MEMORY_SIZE:]

    # Step 1 — Retrieve with intent detection
    chunks, intent_data = retrieve(query, namespace=namespace, top_k=top_k, history=trimmed_history)
    intent = intent_data["intent"]

    # Step 2 — Rewrite query with intent awareness
    rewritten = rewrite_query(query, trimmed_history, intent)

    # Step 3 — Generate intent-aware answer
    answer = generate_answer(query, chunks, trimmed_history, intent)

    # Step 4 — Extract unique sources
    sources = list(dict.fromkeys([c["url"] for c in chunks]))

    print(f"\n💬 Answer: {answer}")
    print(f"\n📚 Sources: {sources}")

    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history, sources
